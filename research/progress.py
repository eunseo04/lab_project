"""Shared durable task journal and resource accounting; never changes policy inputs."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import threading
import time
import zlib


LEGACY_GPU_AUTHORIZATION = {'scope': 'cage4-research', 'gpu_hour_cap': 8.0}


def validate_budget_limits(gpu_hours, wall_seconds, authorization, prior_gpu_hours=0.0):
    cap = authorization.get('gpu_hour_cap')
    scope = authorization.get('scope')
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError('GPU authorization needs an explicit ledger scope')
    for value in (gpu_hours, wall_seconds, prior_gpu_hours, cap):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('GPU reservation, authorization, carry-in and wall cap must be finite')
    if gpu_hours < 0 or wall_seconds <= 0 or prior_gpu_hours < 0 or cap <= 0:
        raise ValueError('Positive authorization and wall cap and nonnegative GPU usage are required')
    if gpu_hours + prior_gpu_hours > cap + 1e-9:
        raise ValueError(f'Prior usage plus the campaign reservation exceeds the {cap:g} GPU-hour authorization for {scope}')


class BudgetExceeded(RuntimeError):
    pass


class IncompleteOptimization(RuntimeError):
    pass


class OptimizerPreflightRequired(IncompleteOptimization):
    """A saved training batch is waiting for its measured cost reservation."""


class TaskInProgress(RuntimeError):
    """Another live owner holds this logical request; never submit it twice."""
    pass


def _default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False, default=_default)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def optimizer_batch_identity(pending):
    return digest([pending['trace_hash'],pending['resume_identity']])


def implementation_identity(root):
    """Hash the actual shared actor/observer and imported simulator, not a revision copy."""
    root=Path(root).resolve()
    paths=list((root/'research').glob('*.py'))+list((root/'acd').glob('*.py'))+[root/'research.sh']
    simulator=sys.modules.get('CybORG')
    if simulator is not None:
        paths+=list(Path(simulator.__file__).resolve().parent.rglob('*.py'))
    return {str(p.resolve().relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def _pack(value):
    # Storage must retain dictionary insertion order: nested observations are
    # rendered with the original json.dumps order inside exact SG prompts.
    # Canonical sorting is for identity hashes only, never payload round-trips.
    return zlib.compress(json.dumps(value,ensure_ascii=False,separators=(',', ':'),
        allow_nan=False,default=_default).encode())


def _unpack(value):
    return json.loads(zlib.decompress(value)) if value is not None else None


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=_default)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def rng_snapshot():
    state = {'python': random.getstate()}
    if 'numpy' in sys.modules:
        state['numpy'] = sys.modules['numpy'].random.get_state()
    if 'torch' in sys.modules:
        state['torch_cpu'] = sys.modules['torch'].get_rng_state().tolist()
    return json.loads(canonical(state))


def rng_restore(state):
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    random.setstate(tuples(state['python']))
    if 'numpy' in state:
        import numpy as np
        v = state['numpy']; np.random.set_state((v[0], np.asarray(v[1], dtype=np.uint32), v[2], v[3], v[4]))
    if 'torch_cpu' in state:
        import torch
        torch.set_rng_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8))


class Journal:
    """SQLite transactions keep completed calls and optimizer boundaries durable.

    Logical stochastic calls include their seed and context in the key. A request
    whose transport completion is unknown stays retryable, never becomes a rejection.
    Compressed disk payloads are restored exactly; compression does not reduce tokens.
    """
    def __init__(self, path):
        self.path = Path(path).resolve(); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA busy_timeout=60000')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                context BLOB NOT NULL, request BLOB, response BLOB, metrics BLOB,
                attempts INTEGER NOT NULL DEFAULT 0, owner TEXT, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, kind TEXT NOT NULL,
                status TEXT NOT NULL, context BLOB NOT NULL, metrics BLOB, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS checkpoints (key TEXT PRIMARY KEY, value BLOB NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS values_cache (key TEXT PRIMARY KEY, value BLOB NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS budget_sessions (
                id TEXT PRIMARY KEY, pid INTEGER NOT NULL, start REAL NOT NULL,
                last REAL NOT NULL, stop REAL, gpu_count INTEGER NOT NULL, metadata BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS budget_limits (
                id INTEGER PRIMARY KEY CHECK(id=1), gpu_hours REAL NOT NULL,
                wall_seconds REAL NOT NULL, created REAL NOT NULL);
        ''')
        self.db.commit()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback(); raise

    def close(self):
        with self.lock:
            self.db.close()

    def task(self, task_id):
        with self.lock:
            row = self.db.execute('SELECT kind,status,context,request,response,metrics,attempts,owner FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is None:
            return None
        return dict(zip(('kind','status','context','request','response','metrics','attempts','owner'),
                        [row[0],row[1],*[_unpack(x) for x in row[2:6]],row[6],row[7]]))

    def plan(self, task_id, kind, context, request=None):
        now = time.time()
        with self.transaction() as db:
            db.execute('INSERT OR IGNORE INTO tasks(id,kind,status,context,request,updated) VALUES(?,?,?,?,?,?)',
                       (task_id,kind,'planned',_pack(context),_pack(request),now))

    def event(self, task_id, kind, status, context=None, metrics=None):
        with self.transaction() as db:
            db.execute('INSERT INTO events(task_id,kind,status,context,metrics,created) VALUES(?,?,?,?,?,?)',
                       (task_id,kind,status,_pack(context or {}),_pack(metrics or {}),time.time()))

    def start(self, task_id, kind, context, request=None):
        self.plan(task_id,kind,context,request)
        owner = f'{os.getpid()}:{threading.get_ident()}'
        with self.transaction() as db:
            row = db.execute('SELECT status,owner FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row[0] == 'completed':
                return False
            if row[0] == 'started' and row[1]:
                try:
                    pid = int(row[1].split(':')[0]); os.kill(pid, 0)
                except ProcessLookupError:
                    pass  # The dead owner's completion is unknown; retry explicitly.
                except (PermissionError, ValueError):
                    raise TaskInProgress(task_id)
                else:
                    raise TaskInProgress(task_id)
            db.execute('UPDATE tasks SET status=?,attempts=attempts+1,owner=?,updated=? WHERE id=?',
                       ('started',owner,time.time(),task_id))
            db.execute('INSERT INTO events(task_id,kind,status,context,metrics,created) VALUES(?,?,?,?,?,?)',
                       (task_id,kind,'started',_pack(context),_pack({}),time.time()))
        return True

    def finish(self, task_id, response, metrics=None):
        with self.transaction() as db:
            row = db.execute('SELECT kind,context FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row is None:
                raise KeyError('Task must be registered before completion')
            db.execute('UPDATE tasks SET status=?,response=?,metrics=?,updated=? WHERE id=?',
                       ('completed',_pack(response),_pack(metrics or {}),time.time(),task_id))
            db.execute('INSERT INTO events(task_id,kind,status,context,metrics,created) VALUES(?,?,?,?,?,?)',
                       (task_id,row[0],'completed',row[1],_pack(metrics or {}),time.time()))

    def mark(self, task_id, status, metrics=None):
        if status not in ('failed','interrupted','unknown','deferred'):
            raise ValueError(status)
        with self.transaction() as db:
            row = db.execute('SELECT kind,context FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            db.execute('UPDATE tasks SET status=?,metrics=?,updated=? WHERE id=?',
                       (status,_pack(metrics or {}),time.time(),task_id))
            db.execute('INSERT INTO events(task_id,kind,status,context,metrics,created) VALUES(?,?,?,?,?,?)',
                       (task_id,row[0],status,row[1],_pack(metrics or {}),time.time()))

    def checkpoint(self, key, value):
        with self.transaction() as db:
            db.execute('INSERT INTO checkpoints(key,value,updated) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated=excluded.updated',
                       (key,_pack(value),time.time()))

    def restore(self, key):
        with self.lock:
            row = self.db.execute('SELECT value FROM checkpoints WHERE key=?',(key,)).fetchone()
        return _unpack(row[0]) if row else None

    def cache_get(self, key):
        with self.lock:
            row = self.db.execute('SELECT value FROM values_cache WHERE key=?',(key,)).fetchone()
        return _unpack(row[0]) if row else None

    def cache_put(self, key, value, replace=False):
        with self.transaction() as db:
            conflict='DO UPDATE SET value=excluded.value,updated=excluded.updated' if replace else 'DO NOTHING'
            db.execute('INSERT INTO values_cache(key,value,updated) VALUES(?,?,?) ON CONFLICT(key) '+conflict,
                       (key,_pack(value),time.time()))

    def tasks(self, statuses=None):
        with self.lock:
            rows = self.db.execute('SELECT id,kind,status,context,metrics,attempts FROM tasks ORDER BY rowid').fetchall()
        return [{'id':r[0],'kind':r[1],'status':r[2],'context':_unpack(r[3]),'metrics':_unpack(r[4]),'attempts':r[5]}
                for r in rows if statuses is None or r[2] in statuses]

    def events(self):
        with self.lock:
            rows = self.db.execute('SELECT seq,task_id,kind,status,context,metrics,created FROM events ORDER BY seq').fetchall()
        return [dict(zip(('seq','task_id','kind','status','context','metrics','created'),
                         [r[0],r[1],r[2],r[3],_unpack(r[4]),_unpack(r[5]),r[6]])) for r in rows]

    def budget_configure(self, gpu_hours, wall_seconds, now=None, *, authorization=None, prior_gpu_hours=0.0):
        authorization = dict(LEGACY_GPU_AUTHORIZATION if authorization is None else authorization)
        validate_budget_limits(gpu_hours, wall_seconds, authorization, prior_gpu_hours)
        if gpu_hours == 0:
            raise ValueError('A positive campaign GPU reservation is required to start a session')
        provenance = {'authorization': authorization, 'prior_gpu_hours': prior_gpu_hours}
        now = time.time() if now is None else now
        with self.transaction() as db:
            previous = db.execute('SELECT gpu_hours,wall_seconds,created FROM budget_limits WHERE id=1').fetchone()
            if previous and (previous[0],previous[1]) != (gpu_hours,wall_seconds):
                raise ValueError('Resume cannot reset or silently change registered budget limits')
            registered = db.execute("SELECT value FROM checkpoints WHERE key='budget_authorization'").fetchone()
            if registered and _unpack(registered[0]) != provenance:
                raise ValueError('Resume cannot change the GPU authorization or its carry-in')
            if previous and not registered and authorization != LEGACY_GPU_AUTHORIZATION:
                raise ValueError('An existing legacy journal cannot be moved to another GPU authorization')
            if previous is None:
                db.execute('INSERT INTO budget_limits VALUES(1,?,?,?)',(gpu_hours,wall_seconds,now))
            if not registered:
                db.execute('INSERT INTO checkpoints VALUES(?,?,?)',('budget_authorization',_pack(provenance),now))

    def budget_start(self, session_id, gpu_count, metadata=None, now=None):
        if gpu_count < 1:
            raise ValueError('At least one occupied GPU is required')
        now = time.time() if now is None else now
        with self.transaction() as db:
            db.execute('INSERT INTO budget_sessions VALUES(?,?,?,?,?,?,?)',
                       (session_id,os.getpid(),now,now,None,gpu_count,_pack(metadata or {})))

    def budget_heartbeat(self, session_id, stop=False, now=None):
        now = time.time() if now is None else now
        with self.transaction() as db:
            db.execute('UPDATE budget_sessions SET last=?,stop=CASE WHEN ? THEN ? ELSE stop END WHERE id=? AND stop IS NULL',
                       (now,stop,now,session_id))

    def budget_usage(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            limit = self.db.execute('SELECT gpu_hours,wall_seconds,created FROM budget_limits WHERE id=1').fetchone()
            authorization = self.db.execute("SELECT value FROM checkpoints WHERE key='budget_authorization'").fetchone()
            rows = self.db.execute('SELECT id,pid,start,last,stop,gpu_count,metadata FROM budget_sessions ORDER BY start').fetchall()
        seconds = sum(((r[4] if r[4] is not None else now)-r[2])*r[5] for r in rows)
        return {'gpu_hours':seconds/3600,'gpu_hour_cap':limit[0] if limit else None,
                'wall_seconds':now-limit[2] if limit else 0,'wall_second_cap':limit[1] if limit else None,
                'budget_authorization':_unpack(authorization[0]) if authorization else None,
                'sessions':[{'id':r[0],'pid':r[1],'start':r[2],'heartbeat':r[3],'stop':r[4],'gpu_count':r[5],'metadata':_unpack(r[6])} for r in rows]}

    def check_budget(self, now=None):
        usage = self.budget_usage(now)
        if usage['gpu_hour_cap'] is not None and (usage['gpu_hours'] >= usage['gpu_hour_cap'] or usage['wall_seconds'] >= usage['wall_second_cap']):
            raise BudgetExceeded('Registered GPU-hour or wall-clock cap reached')
        return usage
