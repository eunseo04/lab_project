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


def serve(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fastapi import FastAPI, HTTPException
    import uvicorn
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
    identity = {"model": args.model_id, "revision": args.revision,
        "backend": "transformers", "transformers": __import__("transformers").__version__,
        "torch": torch.__version__, "dtype": "bfloat16", "weights_frozen": not any(p.requires_grad for p in model.parameters()),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "temperature_scoring": True, "scoring_mode":"incremental_teacher_forcing",
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    app = FastAPI()

    def prompt_ids(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        return text, ids

    @app.get("/health")
    def health():
        return identity

    @app.post("/generate")
    def generate(body: dict):
        started = time.perf_counter()
        temperature = float(body.get("temperature", 1.0))
        if temperature <= 0:
            raise HTTPException(400, "Positive temperature required for PPO probability support")
        text, ids = prompt_ids(body["messages"])
        maximum = int(body.get("max_tokens", 192))
        if len(ids) + maximum > args.max_context:
            raise HTTPException(400, f"Context overflow: {len(ids)}+{maximum}>{args.max_context}; input was not truncated")
        with lock, torch.inference_mode():
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
        return {"text": tokenizer.decode(target, skip_special_tokens=True), "token_ids": target,
            "prompt_tokens": len(ids), "output_tokens": len(target), "generation_logprob": logprob,
            "stopped_by_eos": bool(target and target[-1] == tokenizer.eos_token_id),
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "seconds": time.perf_counter() - started, "identity": identity}

    @app.post("/score")
    def score(body: dict):
        temperature = float(body.get("temperature", 1.0))
        if temperature <= 0:
            raise HTTPException(400, "Positive temperature required")
        results = []
        with lock, torch.inference_mode():
            for sample in body["samples"]:
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
                results.append({"sum": value, "mean": value / len(target), "tokens": len(target)})
        return {"scores": results, "identity": identity}

    @app.post("/embed")
    def embed(body: dict):
        nonlocal encoder
        with lock:
            if encoder is None:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(args.encoder, device="cpu", local_files_only=True)
            vectors = encoder.encode(body["texts"], normalize_embeddings=True)
        return {"vectors": vectors.tolist(), "model": "sentence-transformers/all-MiniLM-L6-v2"}

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--encoder", required=True)
    p.add_argument("--revision")
    p.add_argument("--port", type=int, default=8018)
    p.add_argument("--max-context", type=int, default=24576)
    args=p.parse_args()
    if args.revision is None:
        if args.model_id != "Qwen/Qwen3-8B":p.error("A different model requires an explicit --revision")
        args.revision="b968826d9c46dd6066d109eabc6255188de91218"
    serve(args)
