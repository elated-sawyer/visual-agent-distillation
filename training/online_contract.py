"""Functions retained verbatim from the validated online adapter."""
import copy, hashlib, json, random
from pathlib import Path
def read(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def validate_catalogue(catalogue):
    if digest(catalogue['protocol']) != catalogue['protocol_sha256']:
        raise ValueError('Environment protocol changed')
    tasks = catalogue['tasks']
    if not tasks or any(t['split'] != 'train' for t in tasks):
        raise ValueError('Online collection requires a nonempty training-only catalogue')
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Catalogue must contain unique task identities')
    if any(type(t['seed']) is not int for t in tasks):
        raise ValueError('Each task requires its original reset seed')

def draw_tasks(catalogue, *, seed, rollout_id, wave, count):
    validate_catalogue(catalogue)
    rng = random.Random(f'{seed}:online-tasks:{rollout_id}:{wave}')
    return [rng.choice(catalogue['tasks']) for _ in range(count)]
