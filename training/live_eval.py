"""Blocking native Slime evaluation against frozen, explicitly scoped panels."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from evaluation_contract import validate_panel, validate_schedule

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)

def get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as response:
        return json.load(response)

def identity(urls, expected, context):
    result = {}
    for url in urls:
        version = str(get(url + '/get_weight_version')['weight_version'])
        assert version == str(expected), (url, version, expected)
        info = get(url + '/get_server_info')
        assert info['max_req_input_len'] >= context and not info.get('allow_auto_truncate', False)
        assert any(m['id'] == 'mm-student' for m in get(url + '/v1/models')['data'])
        result[url] = {'weight_version': version, 'capacity': info['max_req_input_len']}
    return result

def generate(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.base_types import RolloutFnEvalOutput
    assert evaluation and not args.debug_train_only
    run = Path(os.environ['MM_OPD_RUN']); cfg = read(run / 'config.json'); step = rollout_id + 1
    validate_schedule(cfg)
    item = next(x for x in cfg['curve_evaluations'] if x['step'] == step)
    workers = get(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/workers')['workers']
    urls = sorted(w['url'].rstrip('/') for w in workers)
    assert len(urls) == len(set(urls)) == cfg['rollout_gpus']
    assert all(url.startswith('http://' + cfg['inference_ip'] + ':') for url in urls)
    before = identity(urls, rollout_id + 2, cfg['evaluation_context_limit'])
    datasets = {}; episodes = 0
    for panel in item['panels']:
        root = run / 'live-evaluations' / f"{step:04d}-{panel['name']}"
        assert not root.exists(), 'Refuse duplicate evaluation'
        request = {'run': str(run), 'step': step, 'rollout_id': rollout_id,
                   'expected_version': str(rollout_id + 2), 'evaluation': panel['eval'],
                   'engines': urls, 'before': before, 'panel': panel,
                   'eval_source_sha256': hashlib.sha256((Path(panel['eval']) / 'manifest.json').read_bytes()).hexdigest(),
                   'environment': cfg['curve_environment'], 'time': time.time()}
        save(root / 'request.json', request)
        deadline = time.monotonic() + cfg.get('evaluation_timeout_seconds', 43200)
        while not (root / 'result.json').exists():
            for failure in (root / 'failed.json', Path(cfg['render_queue']) / 'failed.json'):
                if failure.exists():
                    raise RuntimeError(f'Evaluation failed: {failure}: {read(failure)}')
            if time.monotonic() > deadline:
                raise TimeoutError(str(root))
            time.sleep(2)
        result = read(root / 'result.json')
        digest = hashlib.sha256((root / 'request.json').read_bytes()).hexdigest()
        assert result['status'] == 'passed' and result['request_sha256'] == digest
        assert result['step'] == step and result['expected_version'] == str(rollout_id + 2)
        after = identity(urls, rollout_id + 2, cfg['evaluation_context_limit'])
        assert before == after
        values = validate_panel(panel, result['records'])
        assert not datasets.keys() & values.keys()
        datasets.update({k: {'rewards': v} for k, v in values.items()})
        episodes += len(result['records'])
        save(root / 'acknowledged.json', {'status': 'passed', 'step': step, 'before': before,
             'after': after, 'same_live_learner': True, 'request_sha256': digest, 'time': time.time()})
    return RolloutFnEvalOutput(data=datasets, metrics={'completed_updates': step, 'episodes': episodes})
