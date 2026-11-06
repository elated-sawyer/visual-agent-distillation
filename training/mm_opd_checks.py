"""Validation hooks around native Slime OPD; no loss or gradient replacement."""
import copy, hashlib, json, math, os
from pathlib import Path
import torch
import torch.distributed as dist
from slime.rollout.on_policy_distillation import post_process_rewards as native_postprocess
from mm_opd_transport import exact_reward
from mosaic_distill.scoring import response_logprobs


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


async def reward_func(args,sample,**kwargs):
    assert sample.metadata['task']['split']=='train'
    assert sample.metadata['source_role']=='student'
    assert sample.multimodal_inputs['images']
    assert 0 < sample.response_length <=args.rollout_max_response_len
    assert len(sample.tokens)<=args.rollout_max_context_len
    result=await exact_reward(args,sample,**kwargs)
    try:
        values=response_logprobs(result,sample.tokens[-sample.response_length:])
    except ValueError:
        save(Path(os.environ['MM_OPD_RUN'])/'scoring-failures'/f'{sample.index}.json',{'input_ids':sample.tokens,'response_length':sample.response_length,'result':result})
        raise
    assert len(values)==sample.response_length
    assert len(result['meta_info']['input_token_logprobs'])==len(sample.tokens)
    sample.metadata['teacher_exact_response_ids_verified']=True
    if hasattr(args, 'mm_formal_run') and args.mm_formal_run:
        save(Path(os.environ['MM_OPD_RUN'])/'responses'/f'{sample.index}.json', {
            'sample_index': sample.index, 'metadata': sample.metadata,
            'response': sample.response, 'response_tokens': sample.response_length,
            'total_tokens': len(sample.tokens), 'images': len(sample.multimodal_inputs['images']),
            'weight_versions': sample.weight_versions, 'tokens': sample.tokens,
            'teacher_response_log_probs': values})
    # Preserve the entire teacher payload; standard metrics read only this scalar.
    result['opd_task_reward']=0.0
    return result


def post_process_rewards(args,samples,**kwargs):
    assert args.reward_key=='opd_task_reward'
    teacher_args=copy.copy(args)
    teacher_args.reward_key=None
    result=native_postprocess(teacher_args,samples,**kwargs)
    for sample in samples:
        assert sample.metadata['teacher_exact_response_ids_verified']
        assert len(sample.teacher_log_probs)==sample.response_length
        assert torch.isfinite(sample.teacher_log_probs).all()
    return result


def audit_signal(args,rollout_id,data):
    assert args.opd_kl_coef==1 and not args.normalize_advantages
    assert args.kl_coef==0 and args.kl_loss_coef==0 and not args.use_kl_loss
    rows=[]
    for teacher,old,adv,mask,n in zip(data['teacher_log_probs'],data['log_probs'],data['advantages'],data['loss_masks'],data['response_lengths'],strict=True):
        teacher=teacher.to(old.device)
        assert len(old)==len(teacher)==len(adv)==len(mask)==n
        assert torch.isfinite(old).all() and torch.isfinite(adv).all()
        assert torch.all(mask==1), 'Expected response-only native masks'
        error=(adv-(teacher-old)).abs().max().item()
        assert error<1e-5
        rows.append({'response_tokens':n,'signal_abs_mean':adv.abs().mean().item(),'signal_error_max':error})
    if 'rollout_log_probs' in data:
        for row, old, generated in zip(rows, data['log_probs'], data['rollout_log_probs'], strict=True):
            generated=torch.as_tensor(generated,device=old.device)
            assert torch.isfinite(generated).all() and len(old)==len(generated)
            gap=(old-generated).abs()
            row.update(old_rollout_gap_mean=gap.mean().item(),old_rollout_gap_max=gap.max().item())
    assert all(float(x)==0 for x in data['rewards'])
    save(Path(os.environ['MM_OPD_RUN'])/'signal'/f'{rollout_id}-rank{dist.get_rank()}.json',{'status':'passed','rollout_id':rollout_id,'rank':dist.get_rank(),'samples':rows})


def weights_before_step(args,rollout_id,step_id,model,optimizer,opt_param_scheduler):
    rows=[]
    for part in model:
        names=[]
        for name,param in part.named_parameters():
            if ('visual' in name or 'word_embeddings' in name or 'linear_qkv' in name) and param.numel()>0:
                # Hash finite parameter slices to observe actual updates without copying a model.
                flat=param.detach().flatten()
                count=min(flat.numel(),4096)
                positions=torch.arange(count,dtype=torch.int64,device=flat.device)
                positions=positions*(flat.numel()-1)//max(count-1,1)
                values=flat[positions].float().cpu()
                assert torch.isfinite(values).all()
                names.append({'name':name,'sha256':hashlib.sha256(values.numpy().tobytes()).hexdigest()})
        rows.extend(names)
    assert rows
    save(Path(os.environ['MM_OPD_RUN'])/'weights'/f'{rollout_id}-{step_id}-rank{dist.get_rank()}.json',{'parameters':rows,'rollout_id':rollout_id,'step_id':step_id,'rank':dist.get_rank()})
