"""Environment adapter for Slime's native background worker; no custom queue."""
import asyncio
import copy
import os
import threading
import time
from pathlib import Path

from online_contract import digest, read
from turn_stream_contract import check_execution, save, validate_request

SOURCE = None


class TurnDataSource:
    """Admit environment decisions, with one batch of credit and a finite budget."""

    def __init__(self, args):
        global SOURCE
        self.args = args
        self.run = Path(os.environ['MM_OPD_RUN'])
        self.root = self.run / 'turn-stream'
        self.cfg = read(self.run / 'config.json')
        self.catalogue = read(self.run / 'online-catalogue.json')
        self.consumers = {}
        self.lock = threading.Lock()
        self.busy_lanes = set()
        self.issued = self.accepted = self.batches = 0
        self.total = args.num_rollout * args.rollout_batch_size
        assert args.rollout_batch_size == 64 and args.n_samples_per_prompt == 1
        assert args.rollout_data_transport == 'object-store'
        SOURCE = self

    def __len__(self):
        return self.total

    def load(self, rollout_id):
        assert rollout_id in (None, -1, 0), 'The persistent stream cannot resume a model'

    def save(self, rollout_id):
        raise RuntimeError('The persistent stream uses live evaluation, not checkpoint resume')

    def get_samples(self, count):
        from slime.utils.types import Sample
        from turn_stream_rollout import ready_decisions
        assert count == 1
        with self.lock:
            if self.issued >= min(self.total, self.accepted + 64):
                return []
            for path, item in ready_decisions(self.root):
                if item['lane_id'] in self.busy_lanes:
                    continue
                if (path / 'reserved.json').exists():
                    continue
                validate_request(item, self.catalogue, self.args.seed)
                index = self.issued
                save(path / 'reserved.json', {'sample_index': index, 'time': time.time()})
                self.issued += 1
                self.busy_lanes.add(item['lane_id'])
                return [[Sample(index=index, group_index=index, rollout_id=index,
                                prompt='', metadata={'decision': str(path)})]]
        return []

    def add_samples(self, samples):
        raise RuntimeError('Unexpected abort: turns must drain before weight publication')

    def close(self):
        for worker in self.consumers.values():
            worker.pause(drain=True)
            worker.close()


async def generate_turn(args, original, sampling_params):
    from mosaic_distill.assets import ImageStore
    from slime.rollout.sglang_rollout import GenerateState, generate
    from live_eval import get
    from online_rollout import native_messages, prepare_sample

    source = SOURCE
    path = Path(original.metadata['decision'])
    start = time.time()
    while True:
        item = read(path / 'request.json')
        validate_request(item, source.catalogue, args.seed)
        workers = get(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/workers')['workers']
        versions = {str(get(w['url'].rstrip('/') + '/get_weight_version')['weight_version']) for w in workers}
        assert len(versions) == 1
        version = versions.pop()
        assert version.isdigit() and int(version) >= 1
        assert all(1 <= int(v) <= int(version) for v in item['history_versions'])
        claim = {'rollout_id': int(version) - 1, 'expected_version': version,
                 'sample_index': original.index, 'request_sha256': digest(item)}
        state = GenerateState(args)
        sample = prepare_sample(state, native_messages(item['messages'], ImageStore(source.root / 'assets')), original.index)
        sample.rollout_id = original.rollout_id
        available = min(args.rollout_max_response_len, args.rollout_max_context_len - len(sample.tokens))
        if available <= 0:
            assert item['turn_index'] > 0
            claim['sample_index'] = None
            save(path / 'claim.json', claim)
            response = {**claim, 'status': 'context_limit'}
            save(path / 'response.json', response)
            executed = await wait_execution(source, path)
            check_execution(item, claim, response, executed)
            # Same reserved credit follows the reset; no generated response is discarded.
            lane = item['lane_id']
            deadline = time.monotonic() + source.cfg['online_request_timeout_seconds']
            while True:
                pointer = read(source.root / 'lanes' / f'{lane:02d}' / 'ready.json')
                if pointer['sequence'] > item['sequence']:
                    from turn_stream_contract import decision_dir
                    path = decision_dir(source.root, lane, pointer['sequence'])
                    save(path / 'reserved.json', {'sample_index': original.index, 'time': time.time()})
                    break
                assert time.monotonic() < deadline
                await asyncio.sleep(.1)
            continue
        save(path / 'claim.json', claim)
        params = sampling_params.copy()
        params['max_new_tokens'] = available
        # The native generate_and_rm already owns the generation semaphore.
        sample = await generate(args, sample, params)
        assert sample.status.value in ('completed', 'truncated') and sample.response_length > 0
        assert sample.weight_versions == [version] and len(sample.tokens) <= args.rollout_max_context_len
        assert len(sample.rollout_log_probs) == sample.response_length
        response = {**claim, 'status': 'passed', 'response': sample.response,
                    'response_tokens': sample.response_length, 'prompt_tokens': len(sample.tokens) - sample.response_length}
        save(path / 'response.json', response)
        executed = await wait_execution(source, path)
        check_execution(item, claim, response, executed)
        prefix = {'schema': 'mm-online-decision-v1', 'task': item['task'], 'turn_index': item['turn_index'],
                  'protocol_sha256': source.catalogue['protocol_sha256'], 'messages': item['messages']}
        sample.metadata = {'task': item['task'], 'source_role': 'student', 'prefix_sha256': digest(prefix),
                           'online_episode_id': item['episode_id'], 'online_turn': item['turn_index'],
                           'online_rollout_id': int(version) - 1, 'draw_index': sample.index, 'pool_index': sample.index,
                           'collection_index': item['episode_id'], 'history_versions': item['history_versions'],
                           'sampling_weight_version': version, 'executed_request_sha256': digest(item),
                           'available_response_tokens': available, 'decision': str(path),
                           'generation_started': start, 'execution_finished': time.time()}
        with source.lock:
            source.busy_lanes.remove(item['lane_id'])
        return sample


async def wait_execution(source, path):
    deadline = time.monotonic() + source.cfg['online_request_timeout_seconds']
    while not (path / 'executed.json').exists():
        if (source.root / 'failed.json').exists():
            raise RuntimeError('Persistent environment failed')
        assert time.monotonic() < deadline, 'Environment execution timeout'
        await asyncio.sleep(.1)
    return read(path / 'executed.json')


async def score_turn(args, sample, **kwargs):
    from mm_opd_checks import reward_func
    source = SOURCE
    if not hasattr(source, 'score_gate'):
        source.score_gate = asyncio.Semaphore(16)
    async with source.score_gate:
        sample.metadata['scoring_started'] = time.time()
        result = await reward_func(args, sample, **kwargs)
    save(source.run / 'async-audit' / 'responses' / f'{sample.index:04d}.json',
         {'sample_index': sample.index, 'weight_version': sample.metadata['sampling_weight_version'],
          'generation_started': sample.metadata['generation_started'],
          'execution_finished': sample.metadata['execution_finished'],
          'scoring_started': sample.metadata['scoring_started'], 'scoring_finished': time.time(),
          'response_tokens': sample.response_length})
    return result


def accept_batch(args, groups):
    """Native batch hook validates all 64 samples; it performs no selection."""
    source = SOURCE
    samples = [s for group in groups for s in group]
    with source.lock:
        batch = source.batches
        assert len(samples) == 64
        assert {s.index for s in samples} == set(range(batch * 64, (batch + 1) * 64))
        lags = []
        rows = []
        for sample in sorted(samples, key=lambda s: s.index):
            version = int(sample.metadata['sampling_weight_version'])
            lag = batch + 1 - version
            assert 0 <= lag <= 1, (batch, sample.index, version)
            assert sample.weight_versions == [str(version)]
            assert sample.metadata['teacher_exact_response_ids_verified']
            assert len(sample.rollout_log_probs) == sample.response_length
            lags.append(lag)
            rows.append({'metadata': copy.deepcopy(sample.metadata)})
        with (source.run / 'prefixes.jsonl').open('a') as handle:
            import json
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        save(source.run / 'async-audit' / 'batches' / f'{batch:04d}.json',
             {'status': 'passed', 'batch': batch, 'responses': 64, 'lags': lags,
              'sample_indices': [s.index for s in samples], 'time': time.time(),
              'response_tokens': sum(s.response_length for s in samples)})
        source.accepted += 64
        source.batches += 1


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.fully_async_rollout import AsyncRolloutWorker, generate_rollout_fully_async
    from slime.utils.http_utils import get_rollout_num_engines

    assert not evaluation and data_source is SOURCE
    if 'fully_async' not in data_source.consumers:
        class EnvironmentWorker(AsyncRolloutWorker):
            def pause(self, *, drain=True):
                # An environment action cannot be aborted/requeued like plain text.
                return super().pause(drain=True)
        save(data_source.root / 'request.json', {'run': str(data_source.run), 'lanes': 16,
                                                'schema': 'persistent-online-turns-v1'})
        worker = EnvironmentWorker(args, data_source,
                                   concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args))
        data_source.consumers['fully_async'] = worker
        worker.start()
    return generate_rollout_fully_async(args, rollout_id, data_source)


def evaluate(args, rollout_id, data_source, evaluation=False):
    from live_eval import generate
    from turn_stream_rollout import ready_decisions
    assert evaluation
    worker = data_source.consumers['fully_async']
    was_paused = worker.pause(drain=True)
    deadline = time.monotonic() + 120
    while len(ready_decisions(data_source.root)) != 16:
        assert time.monotonic() < deadline, 'Environment lanes not idle for evaluation'
        time.sleep(.1)
    save(data_source.root / 'quiescent.json', {'status': 'passed', 'waiting_lanes': 16,
                                            'inflight_requests': 0, 'time': time.time()})
    result = generate(args, rollout_id, data_source, evaluation=True)
    if not was_paused and rollout_id + 1 < args.num_rollout:
        worker.resume()
    return result
