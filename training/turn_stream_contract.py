"""Causal online-turn contracts; no outcome filtering or loss replacement."""
from pathlib import Path
from online_contract import digest, read, draw_tasks, validate_catalogue
import json,os

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)

from mosaic_distill.records import validate_message

def task_for_lane(catalogue, seed, lane, cycle, lanes=16):
    assert 0 <= lane < lanes and cycle >= 0
    return draw_tasks(catalogue, seed=seed, rollout_id=0, wave=cycle, count=lanes)[lane]

def decision_dir(root, lane, sequence):
    assert type(lane) is int and 0 <= lane < 16
    assert type(sequence) is int and sequence >= 0
    return Path(root) / 'decisions' / f'{lane:02d}' / f'{sequence:08d}'

def validate_request(item, catalogue, seed):
    validate_catalogue(catalogue)
    lane, cycle, turn = item['lane_id'], item['episode_cycle'], item['turn_index']
    decision_dir('.', lane, item['sequence'])
    assert item['episode_id'] == cycle * 16 + lane
    assert item['task'] == task_for_lane(catalogue, seed, lane, cycle)
    assert item['protocol_sha256'] == catalogue['protocol_sha256']
    assert type(turn) is int and 0 <= turn < catalogue['protocol']['max_turns']
    messages = item['messages']
    assert len(messages) == 2 * turn + 2
    assert [m['role'] for m in messages] == ['system'] + ['user','assistant'] * turn + ['user']
    for message in messages:
        validate_message(message, role=message['role'])
    assert len(item['history_versions']) == turn
    assert any(p['type'] == 'image_ref' for m in messages for p in m['content'])
    return digest(item)

def validate_response(request, claim, response):
    h = digest(request)
    assert claim['request_sha256'] == response['request_sha256'] == h
    assert claim['rollout_id'] == response['rollout_id']
    assert claim['expected_version'] == response['expected_version'] == str(claim['rollout_id'] + 1)
    assert claim['sample_index'] == response['sample_index']
    assert response['status'] in ('passed','context_limit')
    if response['status'] == 'passed':
        assert type(claim['sample_index']) is int and claim['sample_index'] >= 0
        assert isinstance(response['response'], str) and response['response_tokens'] > 0
    else:
        assert claim['sample_index'] is None

def check_execution(request, claim, response, executed):
    validate_response(request, claim, response)
    assert executed['status'] == 'passed'
    assert executed['request_sha256'] == digest(request)
    assert executed['response_sha256'] == digest(response)
    assert executed['expected_version'] == claim['expected_version']
    assert executed['sample_index'] == claim['sample_index']
    assert executed['action_executed'] == (response['status'] == 'passed')
    assert executed['lane_id'] == request['lane_id']
    assert executed['sequence'] == request['sequence']

class RoundLedger:
    def __init__(self, rollout_id, batch_size=64):
        assert rollout_id >= 0 and batch_size == 64
        self.rollout_id = rollout_id
        self.batch_size = batch_size
        self.keys = set()
        self.claims = []
    def claim(self, item):
        key = (item['lane_id'], item['sequence'])
        assert len(self.claims) < self.batch_size, 'Round credit exhausted'
        assert key not in self.keys, 'Duplicate decision'
        self.keys.add(key)
        value = {'rollout_id':self.rollout_id, 'expected_version':str(self.rollout_id+1),
                 'sample_index':self.rollout_id*64+len(self.claims), 'request_sha256':digest(item)}
        self.claims.append(value)
        return value
