"""Frozen Qwen3-8B local backend. Run with .model-venv/bin/python, independently of CC4.

Teacher forcing scores exact emitted token IDs under the SAME tokenizer, chat
template, temperature and untruncated full-vocabulary sampling distribution.
There is no training, adapter, quantization, top-logprob approximation, or fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
import os
import inspect

from contextlib import contextmanager


def full_sequence_scores(model, prefixes, targets, temperature, pad_token_id):
    """Experimental causal teacher forcing; activation needs a parity certificate."""
    import torch
    if not prefixes or len(prefixes) != len(targets) or any(not p or not t for p,t in zip(prefixes,targets)):
        raise ValueError('Nonempty, aligned prompt and exact target IDs are required')
    rows=[p+t[:-1] for p,t in zip(prefixes,targets)]
    width=max(map(len,rows));device=next(model.parameters()).device
    ids=torch.full((len(rows),width),pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        ids[i,-len(row):]=torch.tensor(row,device=device);mask[i,-len(row):]=1
    positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0)
    kwargs={'input_ids':ids,'attention_mask':mask,'position_ids':positions,'use_cache':False}
    if 'logits_to_keep' in inspect.signature(model.forward).parameters:
        kwargs['logits_to_keep']=max(map(len,targets))
    with torch.inference_mode():
        logits=model(**kwargs).logits
        result=[]
        for i,target in enumerate(targets):
            logp=torch.log_softmax(logits[i,-len(target):].float()/temperature,dim=-1)
            selected=logp.gather(1,torch.tensor(target,device=device).unsqueeze(1)).squeeze(1)
            result.append({'sum':selected.sum().item(),'mean':selected.mean().item(),
                           'tokens':len(target),'token_logprobs':selected.tolist()})
    return result


def serve(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fastapi import FastAPI, HTTPException
    import uvicorn
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from research.progress import Journal,digest,TaskInProgress
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").to("cuda:0")
    model.eval()
    model.requires_grad_(False)
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    if hasattr(model.generation_config, "min_p"):
        model.generation_config.min_p = None
    lock = threading.Lock()
    encoder = None
    journal=Journal(args.journal) if getattr(args,'journal',None) else None
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    identity = {"model": args.model_id, "revision": args.revision,
        "backend": "transformers", "transformers": __import__("transformers").__version__,
        "torch": torch.__version__, "dtype": "bfloat16", "weights_frozen": not any(p.requires_grad for p in model.parameters()),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "temperature_scoring": True, "scoring_mode":"incremental_teacher_forcing",
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    identity.update(tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        scorer_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generation_config_sha256=digest(model.generation_config.to_dict()),
        encoder_identity={'model':'sentence-transformers/all-MiniLM-L6-v2','revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41'},
        gpu=gpu,server_pid=os.getpid(),
        scoring_backends=['incremental_reference']+(['full_sequence_experimental'] if getattr(args,'allow_full_sequence',False) else []))
    # Physical placement is telemetry, not part of model-equivalence identity.
    identity.pop('gpu');identity.pop('server_pid')
    app = FastAPI()

    @contextmanager
    def measured_lock(timing):
        waiting=time.perf_counter()
        with lock:
            timing['queue_wait_seconds']=time.perf_counter()-waiting
            torch.cuda.reset_peak_memory_stats()
            began=time.perf_counter();forwards=[];active=[]
            def pre(module,args,kwargs):
                ids=kwargs.get('input_ids',args[0] if args else None)
                event=torch.cuda.Event(enable_timing=True);event.record()
                active.append((event, 'prefill' if ids is not None and ids.shape[-1]>1 else 'decode'))
            def post(module,args,output):
                event,phase=active.pop();end=torch.cuda.Event(enable_timing=True);end.record()
                forwards.append((event,end,phase))
            before=model.register_forward_pre_hook(pre,with_kwargs=True)
            after=model.register_forward_hook(post)
            try:
                yield
            finally:
                before.remove();after.remove();torch.cuda.synchronize()
                timing['forward_calls']=len(forwards)
                for phase in ('prefill','decode'):
                    timing[phase+'_gpu_seconds']=sum(a.elapsed_time(b) for a,b,p in forwards if p==phase)/1000
                timing.update(processing_seconds=time.perf_counter()-began,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=gpu,server_pid=os.getpid())


    def prompt_ids(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        return text, ids

    @app.get("/health")
    def health():
        return identity

    @app.post('/tokenize')
    def tokenize(body: dict):
        began=time.perf_counter();text,ids=prompt_ids(body['messages'])
        return {'prompt_token_ids':ids,'prompt_tokens':len(ids),
                'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'tokenizer_seconds':time.perf_counter()-began,'identity':identity,
                'max_context':args.max_context}

    def durable_generation(function):
        def wrapped(body: dict):
            key='server:'+body['request_id'] if body.get('request_id') else None
            if not journal or key is None:return function(body)
            context=body.get('request_context',{})
            try:
                claimed=journal.start(key,context.get('stage','generation'),context,body)
            except TaskInProgress:
                raise HTTPException(409,'Logical request is running; retry the same request ID')
            if not claimed:return journal.task(key)['response']
            try:
                response=function(body)
                journal.finish(key,response,{**response.get('timing',{}),
                    'input_tokens':response['prompt_tokens'],'output_tokens':response['output_tokens'],
                    'actual_model_calls':1})
                return response
            except BaseException as exc:
                journal.mark(key,'failed',{'error':f'{type(exc).__name__}: {exc}'})
                raise
        return wrapped

    @app.post("/generate")
    @durable_generation
    def generate(body: dict):
        started = time.perf_counter()
        temperature = float(body.get("temperature", 1.0))
        if temperature <= 0:
            raise HTTPException(400, "Positive temperature required for PPO probability support")
        token_start=time.perf_counter()
        text, ids = prompt_ids(body["messages"])
        timing={'tokenizer_seconds':time.perf_counter()-token_start}

        maximum = int(body.get("max_tokens", 192))
        if len(ids) + maximum > args.max_context:
            raise HTTPException(400, f"Context overflow: {len(ids)}+{maximum}>{args.max_context}; input was not truncated")
        with measured_lock(timing), torch.inference_mode():
            torch.manual_seed(int(body.get("seed", 42)))
            inputs = torch.tensor([ids], device="cuda")
            result = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                repetition_penalty=1.0, max_new_tokens=maximum,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=bool(body.get("return_logprobs", False)))
            target = result.sequences[0, len(ids):].tolist()
            logprob = None
            if body.get("return_logprobs"):
                logprob = model.compute_transition_scores(result.sequences, result.scores,
                    normalize_logits=True).sum().item()
        response={"text": tokenizer.decode(target, skip_special_tokens=True), "token_ids": target,
            "prompt_tokens": len(ids), "output_tokens": len(target), "generation_logprob": logprob,
            "stopped_by_eos": bool(target and target[-1] == tokenizer.eos_token_id),
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "seconds": time.perf_counter() - started, "identity": identity,"timing":timing,
            "prompt_token_ids":ids}
        return response

    def durable_score(function):
        def wrapped(body: dict):
            try:return function(body)
            except BaseException as exc:
                if journal:
                    for request_id in body.get('request_ids',[]):
                        key='server:'+request_id;record=journal.task(key)
                        if record and record['status']=='started' and record['owner']==f'{os.getpid()}:{threading.get_ident()}':
                            journal.mark(key,'failed',{'error':f'{type(exc).__name__}: {exc}'})
                raise
        return wrapped

    @app.post("/score")
    @durable_score
    def score(body: dict):
        temperature = float(body.get("temperature", 1.0))
        if temperature <= 0:
            raise HTTPException(400, "Positive temperature required")
        backend=body.get('backend','incremental_reference')
        timing={}
        if backend=='full_sequence_experimental':
            if not getattr(args,'allow_full_sequence',False):
                raise HTTPException(400,'Full-sequence scorer is experimental and disabled')
            original=body['samples'];request_ids=body.get('request_ids',[])
            reused={};missing=[]
            for index,sample in enumerate(original):
                key='server:'+request_ids[index] if len(request_ids)>index else None
                if journal and key:
                    try:claimed=journal.start(key,body.get('request_context',{}).get('stage','score'),body.get('request_context',{}),sample)
                    except TaskInProgress:raise HTTPException(409,'Score already running; retry same request ID')
                    if not claimed:reused[index]=journal.task(key)['response'];continue
                missing.append(index)
            if not missing:return {'scores':[reused[i] for i in range(len(original))],'identity':identity}
            began=time.perf_counter();encoded=[prompt_ids(original[i]['messages'])[1] for i in missing]
            targets=[original[i]['token_ids'] for i in missing]
            if any(len(p)+len(t)>args.max_context for p,t in zip(encoded,targets)):
                raise HTTPException(400,'Scoring context overflow; no truncation is allowed')
            timing['tokenizer_seconds']=time.perf_counter()-began
            try:
                with measured_lock(timing):
                    results=full_sequence_scores(model,encoded,targets,temperature,tokenizer.pad_token_id)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise HTTPException(507,'Full-sequence scorer exceeded memory; use the reference scorer')
            for index,item,prefix in zip(missing,results,encoded):
                item['timing']=timing;reused[index]=item
                if journal and len(request_ids)>index:
                    journal.finish('server:'+request_ids[index],item,{**timing,'target_tokens':item['tokens'],'input_tokens':len(prefix)})
            return {'scores':[reused[i] for i in range(len(original))],'identity':identity}
        if backend!='incremental_reference':raise HTTPException(400,'Unknown scorer')
        results = []
        with measured_lock(timing), torch.inference_mode():
            for sample_index,sample in enumerate(body["samples"]):
                sample_started=time.perf_counter()
                request_ids=body.get('request_ids',[])
                key='server:'+request_ids[sample_index] if len(request_ids)>sample_index else None
                if journal and key:
                    try:claimed=journal.start(key,body.get('request_context',{}).get('stage','score'),
                        body.get('request_context',{}),sample)
                    except TaskInProgress:raise HTTPException(409,'Score already running; retry same request ID')
                    if not claimed:
                        results.append(journal.task(key)['response']);continue
                _, prefix = prompt_ids(sample["messages"])
                target = sample["token_ids"]
                if not target:
                    raise HTTPException(400, "Empty target is not a valid verified action")
                if len(prefix) + len(target) > args.max_context:
                    raise HTTPException(400, "Scoring context overflow; no truncation is allowed")
                from transformers import LogitsProcessor
                captured=[]
                class TeacherForce(LogitsProcessor):
                    def __call__(self,input_ids,scores):
                        index=input_ids.shape[-1]-len(prefix)
                        wanted=target[index]
                        # Capture the ORIGINAL policy probability before forcing
                        # the observed token. This is not the forced distribution.
                        captured.append(torch.log_softmax(scores.float()/temperature,dim=-1)[0,wanted])
                        forced=torch.full_like(scores,-float('inf'))
                        forced[:,wanted]=0
                        return forced
                ids=torch.tensor([prefix],device='cuda')
                teacher=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=True,
                    temperature=temperature,top_p=1.0,top_k=0,repetition_penalty=1.0,
                    max_new_tokens=len(target),pad_token_id=tokenizer.pad_token_id,
                    logits_processor=[TeacherForce()])
                if teacher[0,len(prefix):].tolist()!=target or len(captured)!=len(target):
                    raise HTTPException(500,'Teacher-forced token alignment failed')
                value=torch.stack(captured).sum().item()
                if not __import__("math").isfinite(value):
                    raise HTTPException(500, "Non-finite exact log probability")
                item={"sum": value, "mean": value / len(target), "tokens": len(target),
                      'token_logprobs':[x.item() for x in captured]}
                results.append(item)
                if journal and key:journal.finish(key,item,{'target_tokens':len(target),'actual_model_samples':1,
                    'input_tokens':len(prefix),'processing_seconds':time.perf_counter()-sample_started,
                    'queue_wait_seconds':timing['queue_wait_seconds'] if sample_index==0 else 0,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'gpu':gpu,'server_pid':os.getpid()})

        for item in results:item['timing']=timing
        return {"scores": results, "identity": identity}

    @app.post("/embed")
    def embed(body: dict):
        nonlocal encoder
        timing={}
        with measured_lock(timing):
            if encoder is None:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(args.encoder, device="cpu", local_files_only=True)
            vectors = encoder.encode(body["texts"], normalize_embeddings=True)
        return {"vectors": vectors.tolist(), "model": "sentence-transformers/all-MiniLM-L6-v2","timing":timing}

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--encoder", required=True)
    p.add_argument("--revision")
    p.add_argument("--port", type=int, default=8018)
    p.add_argument("--max-context", type=int, default=24576)
    p.add_argument('--journal',help='Shared durable request database for this campaign')
    p.add_argument('--allow-full-sequence',action='store_true',help='Expose experimental scorer for equivalence profiling only')
    args=p.parse_args()
    if args.revision is None:
        if args.model_id != "Qwen/Qwen3-8B":p.error("A different model requires an explicit --revision")
        args.revision="b968826d9c46dd6066d109eabc6255188de91218"
    serve(args)
