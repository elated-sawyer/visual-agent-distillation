"""Response-only multimodal labels fed to Slime's native SFT loss."""
import hashlib,json,os
from pathlib import Path
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import _prepare_prompt_ids
from slime.utils.processing_utils import load_processor,load_tokenizer
from slime.utils.types import Sample

_tokenizer=None;_processor=None;_prefix_ids=None

def generate_rollout(args,rollout_id,data_source,evaluation=False):
    if evaluation:
        raise ValueError('Teacher-label SFT has no in-loop evaluation; use the frozen environment evaluator')
    global _tokenizer,_processor,_prefix_ids
    run=Path(os.environ['MM_OPD_RUN']);cfg=json.loads((run/'config.json').read_text())
    labels=Path(cfg['labels_run'])
    if _tokenizer is None:
        _tokenizer=load_tokenizer(args.hf_checkpoint,trust_remote_code=True)
        _processor=load_processor(args.hf_checkpoint,trust_remote_code=True)
        _prefix_ids=json.loads((labels/'prefix-token-ids.json').read_text())
    assert args.loss_type=='sft_loss' and not args.compute_advantages_and_returns and not args.use_opd
    groups=data_source.get_samples(args.rollout_batch_size)
    expected_start=rollout_id*args.rollout_batch_size
    records=[]
    for j,group in enumerate(groups):
        assert len(group)==1
        sample=group[0];md=sample.metadata
        assert md['sft_order']==expected_start+j and md['task']['split']=='train'
        label_path=labels/'labels'/f"{md['draw_index']:04d}.json"
        raw=label_path.read_bytes();label=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==md['label_sha256']
        assert label['prefix_sha256']==md['prefix_sha256'] and label['pool_index']==md['pool_index']
        prompt_ids=list(_prepare_prompt_ids(sample,_tokenizer,_processor))
        assert prompt_ids==_prefix_ids[md['pool_index']], 'Teacher/student prefix tokenization differs'
        response=label['response_ids'];assert 0<len(response)<=cfg['response_limit']
        assert len(prompt_ids)+len(response)<=cfg['context_limit']
        assert sample.multimodal_train_inputs and sample.multimodal_inputs['images']
        sample.tokens=prompt_ids+response
        sample.rollout_id=sample.index
        sample.response=label['response_text'];sample.response_length=len(response)
        sample.loss_mask=[1]*len(response);sample.reward=0
        sample.status=Sample.Status.TRUNCATED if label['finish_reason']['type']=='length' else Sample.Status.COMPLETED
        records.append({'sample_index':sample.index,'sft_order':md['sft_order'],'draw_index':md['draw_index'],
          'prefix_sha256':md['prefix_sha256'],'label_sha256':md['label_sha256'],
          'response_tokens':len(response),'prompt_tokens':len(prompt_ids),'images':len(sample.multimodal_inputs['images']),
          'teacher_response_ids_exact':True,'supervised_prompt_tokens':0,'all_response_tokens_supervised':True})
    output=run/'sft-batches';output.mkdir(exist_ok=True)
    (output/f'{rollout_id}.json').write_text(json.dumps({'rollout_id':rollout_id,'samples':records},indent=2)+'\n')
    return RolloutFnTrainOutput(samples=groups)

def audit_batch(args,rollout_id,data):
    import torch
    import torch.distributed as dist
    assert args.loss_type=='sft_loss' and not args.compute_advantages_and_returns
    assert 'advantages' not in data and 'teacher_log_probs' not in data
    lengths=data['response_lengths'];masks=data['loss_masks']
    assert all(len(mask)==n and bool(torch.all(mask==1)) for mask,n in zip(masks,lengths,strict=True))
    assert all(float(r)==0 for r in data['rewards'])
    p=Path(os.environ['MM_OPD_RUN'])/'signal';p.mkdir(exist_ok=True)
    (p/f'{rollout_id}-rank{dist.get_rank()}.json').write_text(json.dumps({'status':'passed','rollout_id':rollout_id,
      'rank':dist.get_rank(),'local_samples':len(lengths),'response_tokens':sum(lengths),'loss':'native_sft_loss','no_advantages':True})+'\n')
