"""Compare Slime's actual native multimodal input paths for every frozen prefix."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor,AutoTokenizer
from slime.utils.data import Dataset
from slime.utils.processing_utils import build_processor_kwargs,process_vision_info
from slime.rollout.sglang_rollout import _prepare_prompt_ids

run=Path(sys.argv[1]);c=json.loads((run/'config.json').read_text())
pool=[json.loads(x) for x in (run/'pool.jsonl').read_text().splitlines()]
draws=[json.loads(x) for x in (run/'prefixes.jsonl').read_text().splitlines()]
assert len(draws)==c['updates']*c['batch_size']
assert len({r['metadata']['episode_sha256'] for r in pool})==c['trajectory_count']
assert {r['metadata']['collection_index'] for r in pool}==set(c['training_indices'])
proc=AutoProcessor.from_pretrained(c['student'],local_files_only=True)
tok=AutoTokenizer.from_pretrained(c['student'],local_files_only=True)
data=Dataset(str(run/'pool.jsonl'),tok,proc,None,prompt_key='messages',apply_chat_template=True,
             apply_chat_template_kwargs={'enable_thinking':True})
checks=[]
for row,sample in zip(pool,data,strict=True):
    ids=list(_prepare_prompt_ids(sample,tok,proc));native=process_vision_info(row['messages'],proc)
    direct=proc(text=sample.prompt,**build_processor_kwargs(native))
    assert ids==list(direct['input_ids'][0])
    for key in ['pixel_values','image_grid_thw']:
        assert torch.equal(direct[key],sample.multimodal_train_inputs[key]),key
    for expected,actual in zip(native['images'],sample.multimodal_inputs['images'],strict=True):
        assert expected.size==actual.size and expected.tobytes()==actual.tobytes()
    assert 0<len(ids) and row['metadata']['task']['split']=='train'
    if len(ids)+c['response_limit']>c['context_limit']:
        (run/'length-failure.json').write_text(json.dumps({'prefix_sha256':row['metadata']['prefix_sha256'],
            'prompt_tokens':len(ids),'response_limit':c['response_limit'],'context_limit':c['context_limit'],
            'automatic_truncation':False}))
        raise ValueError('Prefix exceeds approved context; no truncation or filtering allowed')
    checks.append({'prefix_sha256':row['metadata']['prefix_sha256'],'prompt_tokens':len(ids),
        'images':len(native['images']),'same_native_ids':True,'same_native_pixels':True,
        'same_native_grid':True,'response_allowance':c['response_limit']})
for i,row in enumerate(draws):
    md=row['metadata'];original=pool[md['pool_index']]
    assert md['draw_index']==i and row['messages']==original['messages']
    assert {k:v for k,v in md.items() if k not in ['draw_index','pool_index','sft_order','label_sha256']}==original['metadata']
audit={'status':'passed','pool_samples':len(pool),'tasks':len({r['metadata']['task']['task_id'] for r in pool}),
    'trajectories':c['trajectory_count'],'draws':len(draws),'samples':checks,
    'dataset_sha256':hashlib.sha256((run/'prefixes.jsonl').read_bytes()).hexdigest(),
    'pool_sha256':hashlib.sha256((run/'pool.jsonl').read_bytes()).hexdigest()}
(run/'input-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('NATIVE MULTIMODAL PREFIX AUDIT PASSED',flush=True)
