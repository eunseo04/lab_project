"""Strict frozen-model client with durable exact request/score reuse and telemetry."""
from contextlib import contextmanager
import hashlib
import json
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .progress import BudgetExceeded, Journal, digest


class ModelFailure(RuntimeError):
    pass


class ModelClient:
    is_mock = False

    def __init__(self, config):
        self.config = config
        self.endpoint = config['endpoint'].rstrip('/')
        self.identity = self._request('health')
        for key in ('model', 'revision'):
            if self.identity.get(key) != config[key]:
                raise ModelFailure(f'Wrong local model {key}: expected {config[key]}, got {self.identity.get(key)}')
        if self.identity.get('weights_frozen') is not True:
            raise ModelFailure('The policy backend must certify frozen weights')
        self.calls, self.counter = [], 0
        self._score_cache = OrderedDict()
        self._cache_lock = threading.RLock()
        self._local = threading.local()
        self.base_context = dict(config.get('request_context', {}))
        self.journal = Journal(config['journal_path']) if config.get('journal_path') else None
        self._token_cache = {}
        self._scoring_endpoints = list(dict.fromkeys([self.endpoint] +
            [url.rstrip('/') for url in config.get('scoring_endpoints', [])]))
        self._generation_endpoints = list(dict.fromkeys([self.endpoint] +
            [url.rstrip('/') for url in config.get('generation_endpoints', [])]))
        for endpoint in set(self._scoring_endpoints+self._generation_endpoints)-{self.endpoint}:
            self._validate_scorer(self._request('health', endpoint=endpoint))
        self._generation_workers = queue.Queue()
        for endpoint in self._generation_endpoints:
            self._generation_workers.put(endpoint)
        if config.get('scoring_backend','incremental_reference')!='incremental_reference':
            certificate_path=config.get('scorer_parity_certificate')
            certificate=json.loads(Path(certificate_path).read_text()) if certificate_path else {}
            if (certificate.get('status')!='passed' or certificate.get('model_identity')!=self.identity
                or certificate.get('temperature')!=config.get('temperature',1.0)
                or certificate.get('backend')!=config.get('scoring_backend')
                or not certificate.get('complete_matrix') or not certificate.get('ppo_decisions_match')):
                raise ModelFailure('Experimental scorer requires a complete real-Qwen parity certificate; reference remains the default')
            self.scorer_certificate=certificate

    @contextmanager
    def context(self, **values):
        previous = getattr(self._local, 'context', {})
        self._local.context = {**previous, **values}
        try:
            yield self
        finally:
            self._local.context = previous

    def current_context(self):
        return {**self.base_context, **getattr(self._local, 'context', {})}

    def _request(self, route, payload=None, endpoint=None):
        began=time.perf_counter()
        data = None if payload is None else json.dumps(payload).encode()
        serialized=time.perf_counter()-began
        request = urllib.request.Request((endpoint or self.endpoint)+'/'+route, data=data,
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.config.get('timeout_seconds', 180)) as response:
                result=json.load(response)
                if isinstance(result,dict) and route in ('generate','score','embed','tokenize'):
                    result['transport']={'serialization_seconds':serialized,
                        'http_roundtrip_seconds':time.perf_counter()-began,'request_bytes':len(data or b'')}
                return result
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            detail = exc.read().decode(errors='replace')[:600] if isinstance(exc, urllib.error.HTTPError) else str(exc)
            raise ModelFailure(f'{route} failed: {detail}') from exc

    def _validate_scorer(self, identity):
        required = ('model', 'revision', 'dtype', 'chat_template_sha256', 'scoring_mode')
        if any(identity.get(k) != self.identity.get(k) for k in required) or identity.get('weights_frozen') is not True:
            raise ModelFailure('Scorer weights, tokenizer, dtype or scoring method differ from the acting model')
        for key in ('tokenizer_sha256', 'generation_config_sha256','scorer_implementation_sha256','torch','transformers'):
            if key in self.identity and identity.get(key) != self.identity[key]:
                raise ModelFailure('Replica tokenizer or generation configuration differs')

    def _fingerprint(self):
        return {k: self.identity.get(k) for k in ('model', 'revision', 'dtype', 'chat_template_sha256',
                'tokenizer_sha256', 'generation_config_sha256', 'scoring_mode','scorer_implementation_sha256','torch','transformers')}

    def _tokens(self, messages):
        raw_key = digest({'messages': messages, 'model': self._fingerprint()})
        with self._cache_lock:
            if raw_key in self._token_cache:
                return self._token_cache[raw_key]
        cached = self.journal.cache_get('tokenize:'+raw_key) if self.journal else None
        result = cached or self._request('tokenize', {'messages': messages})
        self._validate_scorer(result['identity'])
        if not result.get('prompt_token_ids'):
            raise ModelFailure('Exact rendered prompt token IDs are required')
        with self._cache_lock:
            self._token_cache[raw_key] = result
        if self.journal and cached is None:
            self.journal.cache_put('tokenize:'+raw_key, result)
        return result

    def _score_key(self, sample, reference=False):
        value = {'model': self._fingerprint(), 'temperature': self.config.get('temperature', 1.0),
                 'scorer': 'incremental_reference' if reference else self.config.get('scoring_backend', 'incremental_reference'),
                 'target_token_ids': sample['token_ids']}
        if self.journal:
            value['prompt_token_ids'] = self._tokens(sample['messages'])['prompt_token_ids']
        else:
            value['messages'] = sample['messages']
        return digest(value)

    def _remember_score(self, key, value):
        with self._cache_lock:
            self._score_cache[key] = value
            self._score_cache.move_to_end(key)
            while len(self._score_cache) > self.config.get('score_cache_size', 8192):
                self._score_cache.popitem(last=False)
        if self.journal:
            self.journal.cache_put('score:'+key, value)

    def _check_budget(self):
        if self.config.get('deadline_unix') and time.time() >= self.config['deadline_unix']:
            raise BudgetExceeded('Request deadline reached')
        if self.journal:
            self.journal.check_budget()

    def generate(self, messages, *, seed=42, purpose='policy', max_tokens=None, return_logprobs=False):
        maximum = max_tokens or self.config.get('max_tokens', 192)
        context = self.current_context()
        stage = context.get('stage') or {'policy':'acting','skill_selection':'selection','termination':'termination'}.get(purpose, purpose)
        payload = {'messages': messages, 'seed': seed, 'temperature': self.config.get('temperature', 1.0),
                   'max_tokens': maximum, 'return_logprobs': return_logprobs}
        task_id, cached, tokenized = None, None, None
        started = time.perf_counter()
        if self.journal:
            tokenized = self._tokens(messages)
            task_id = digest({'context':context,'stage':stage,'model':self._fingerprint(),
                              'prompt_token_ids':tokenized['prompt_token_ids'],'target_token_ids':None,
                              'logical_sampling_seed':seed,'temperature':payload['temperature'],
                              'max_tokens':maximum,'return_logprobs':return_logprobs})
            self.journal.plan(task_id,stage,{**context,'logical_sampling_seed':seed},payload)
            record = self.journal.task(task_id)
            if record and record['status'] == 'completed':
                cached = record['response']
            payload.update(request_id=task_id, request_context={**context,'stage':stage,'logical_sampling_seed':seed})
            if len(tokenized['prompt_token_ids'])+maximum>tokenized.get('max_context',float('inf')):
                self.journal.plan(task_id,stage,context,payload)
                self.journal.mark(task_id,'failed',{'reason':'measured_context_overflow',
                    'input_tokens':len(tokenized['prompt_token_ids']),'reserved_output_tokens':maximum,'actual_model_calls':0})
                raise ModelFailure('Measured context overflow; exact input was not truncated')
        if cached is None:
            self._check_budget()
            if self.journal:
                self.journal.start(task_id,stage,{**context,'logical_sampling_seed':seed},payload)
            endpoint = self._generation_workers.get() if purpose == 'evolution' or context.get('optimizer') else self.endpoint
            leased = purpose == 'evolution' or context.get('optimizer')
            try:
                result = self._request('generate', payload, endpoint=endpoint)
                if any(result.get('identity', {}).get(k) != self.config[k] for k in ('model','revision')):
                    raise ModelFailure('Model identity changed during generation')
                self._validate_scorer(result.get('identity',{}))
                if self.journal:
                    metrics = {**result.get('timing', {}), **result.get('transport',{}),'input_tokens':result['prompt_tokens'],
                               'output_tokens':result['output_tokens'],'endpoint':endpoint,
                               'elapsed_seconds':time.perf_counter()-started,'cache_hit':False,'actual_model_calls':1}
                    self.journal.finish(task_id,result,metrics)
            except BaseException as exc:
                if self.journal:
                    status = 'interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit,BudgetExceeded)) else 'unknown'
                    self.journal.mark(task_id,status,{'error':f'{type(exc).__name__}: {exc}',
                                                   'elapsed_seconds':time.perf_counter()-started})
                raise
            finally:
                if leased:
                    self._generation_workers.put(endpoint)
        else:
            result = cached
            self.journal.event(task_id,stage,'completed',context,{'cache_hit':True,'actual_model_calls':0})
        with self._cache_lock:
            self.counter += 1
            call = {'purpose':purpose,'stage':stage,'input_tokens':result['prompt_tokens'],
                    'output_tokens':result['output_tokens'],'seconds':result['seconds'],
                    'cache_hit':cached is not None,'actual_model_calls':int(cached is None)}
            if purpose in ('skill_selection','termination'):
                call['response'] = result['text']
            self.calls.append(call)
        value = result.get('generation_logprob')
        if value is not None and math.isfinite(value) and self.config.get('scoring_backend','incremental_reference') == 'incremental_reference':
            self._remember_score(self._score_key({'messages':messages,'token_ids':result['token_ids']}),value)
        return result

    def score(self, samples, *, use_cache=True, reference=False):
        if self.identity.get('scoring_mode') != 'incremental_teacher_forcing':
            raise ModelFailure('Exact incremental scoring is unavailable; candidate cannot be verified')
        keys = [self._score_key(sample,reference) for sample in samples]
        cached = {}
        for key in keys:
            if use_cache:
                with self._cache_lock:
                    value = self._score_cache.get(key)
                if value is None and self.journal:
                    value = self.journal.cache_get('score:'+key)
                if value is not None:
                    cached[key] = value
        missing = list(dict.fromkeys(k for k in keys if k not in cached))
        by_key = dict(zip(keys,samples))
        pending = queue.Queue()
        size = max(1,int(self.config.get('score_batch_size',1)))
        for start in range(0,len(missing),size):
            pending.put(missing[start:start+size])
        computed = {}; context = self.current_context(); stage = context.get('stage','score')
        if self.journal:
            for key in missing:
                task_id=digest({'context':context,'stage':stage,'score_key':key,'reference':reference})
                self.journal.plan(task_id,stage,context,by_key[key])
            for key,value in cached.items():
                task_id=digest({'context':context,'stage':stage,'score_key':key,'reference':reference})
                self.journal.plan(task_id,stage,context,by_key[key])
                self.journal.finish(task_id,value,{'cache_hit':True,'actual_model_samples':0,
                    'target_tokens':len(by_key[key]['token_ids'])})
        def worker(endpoint):
            while True:
                try:
                    batch = pending.get_nowait()
                except queue.Empty:
                    return
                task_ids = []
                try:
                    self._check_budget()
                    for key in batch:
                        task_id = digest({'context':context,'stage':stage,'score_key':key,'reference':reference})
                        task_ids.append(task_id)
                        if self.journal:
                            self.journal.start(task_id,stage,context,by_key[key])
                    began = time.perf_counter()
                    response = self._request('score',{'samples':[by_key[k] for k in batch],
                        'temperature':self.config.get('temperature',1.0),
                        'backend':'incremental_reference' if reference else self.config.get('scoring_backend','incremental_reference'),
                        'request_ids':task_ids,'request_context':context},endpoint=endpoint)
                    self._validate_scorer(response.get('identity',{}))
                    if len(response.get('scores',[])) != len(batch):
                        raise ModelFailure('Missing exact teacher-forced score')
                    for key,task_id,item in zip(batch,task_ids,response['scores']):
                        value = item['sum']
                        if not math.isfinite(value):
                            raise ModelFailure('Non-finite exact logprob')
                        if self.journal:
                            self.journal.finish(task_id,value,{**item.get('timing',{}),'endpoint':endpoint,
                                **response.get('transport',{}),
                                'input_tokens':len(self._tokens(by_key[key]['messages'])['prompt_token_ids']),
                                'target_tokens':len(by_key[key]['token_ids']),'elapsed_batch_seconds':time.perf_counter()-began,
                                'cache_hit':False})
                        self._remember_score(key,value)
                        with self._cache_lock:
                            computed[key] = value
                except BaseException as exc:
                    if self.journal:
                        for task_id in task_ids:
                            record = self.journal.task(task_id)
                            if record and record['status'] != 'completed':
                                self.journal.mark(task_id,'interrupted' if isinstance(exc,BudgetExceeded) else 'unknown',
                                                  {'error':f'{type(exc).__name__}: {exc}'})
                    raise
                finally:
                    pending.task_done()
        with ThreadPoolExecutor(max_workers=len(self._scoring_endpoints)) as executor:
            futures = [executor.submit(worker,endpoint) for endpoint in self._scoring_endpoints]
            for future in futures:
                future.result()
        values = [cached[k] if k in cached else computed[k] for k in keys]
        self.calls.append({'purpose':'gate_score','stage':stage,'samples':len(samples),
                           'cache_hits':len(samples)-len(missing),'teacher_forced_samples':len(missing),
                           'replicas':len(self._scoring_endpoints)})
        if self.journal:
            self.journal.event(None,stage,'completed',context,{'samples':len(samples),'cache_hits':len(samples)-len(missing),
                                                             'actual_model_samples':len(missing)})
        return values

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        key = digest({'encoder':self.identity.get('encoder_identity','locked-MiniLM'),'texts':texts})
        vectors = self.journal.cache_get('embed:'+key) if self.journal else None
        if vectors is None:
            vectors = self._request('embed',{'texts':texts})['vectors']
            if self.journal:
                self.journal.cache_put('embed:'+key,vectors)
        if len(vectors) != len(texts):
            raise ModelFailure('Embedding batch alignment failed')
        return vectors


class TextModel:
    """Same sequential sampling seeds, with task-local reserved calls for SG workers."""
    def __init__(self, client):
        self.client = client
        self.seed, self.counter = 42, 0
        self.purpose = 'skill_selection'
        self._lock = threading.Lock()

    def reserve(self, purpose=None):
        with self._lock:
            self.counter += 1
            count = self.counter
        actual = purpose or self.purpose
        seed = int(hashlib.sha256(f'{self.seed}:{count}:{actual}'.encode()).hexdigest()[:8],16)
        return {'seed':seed,'counter':count,'purpose':actual,'max_tokens':768 if self.purpose=='evolution' else 48}

    def invoke(self, prompt, reservation):
        return self.client.generate([{'role':'user','content':prompt}],seed=reservation['seed'],
            purpose=reservation['purpose'],max_tokens=reservation['max_tokens'])['text']

    def __call__(self, prompt, purpose=None):
        return self.invoke(prompt,self.reserve(purpose))

    def score(self, samples, **kwargs):
        return self.client.score(samples,**kwargs)

    def snapshot(self):
        return {'seed':self.seed,'counter':self.counter,'purpose':self.purpose}

    def restore(self, value):
        self.seed,self.counter = value['seed'],value['counter']
        self.purpose = value.get('purpose','evolution')


def stable_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8],16) % (2**31)
