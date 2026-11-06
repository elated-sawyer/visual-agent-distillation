"""Run the native Slime trainer with the published multimodal hooks."""
import hashlib,json,os
from pathlib import Path
from functools import partial

def main():
    import ray
    from slime.utils.arguments import parse_args
    from slime.observability.logging_utils import configure_logger
    from slime.ray.placement_group import create_training_models
    from mm_tracking import TrackingMegatronTrainRayActor
    import train as native_train
    args,restore=parse_args(return_restore_plan=True)
    run=Path(os.environ['MM_OPD_RUN']);cfg=json.loads((run/'config.json').read_text())
    assert Path(args.hf_checkpoint).resolve()==Path(cfg['student']).resolve(), 'Student differs from input-audited model'
    audit=json.loads((run/'input-audit.json').read_text());assert audit['status']=='passed'
    for key,name in [('dataset_sha256','prefixes.jsonl'),('pool_sha256','pool.jsonl')]:
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==audit[key], 'Inputs changed after native validation'
    assert args.num_rollout==cfg['updates'] and args.global_batch_size==args.rollout_batch_size==cfg['batch_size']
    assert args.rollout_max_response_len==cfg['response_limit'] and args.rollout_max_context_len==cfg['context_limit']
    # Native string-only prompt filtering would lose images. Full multimodal
    # lengths, native token identities and image grids were checked above.
    args.rollout_max_prompt_len=None
    args.mm_formal_run=True
    if os.environ.get('MOSAIC_ONLINE')=='1':
        assert args.data_source_path=='native_async_turns.TurnDataSource'
        assert args.use_tis and args.tis_clip==2 and args.tis_clip_low==0
        assert not args.use_rollout_logprobs and not args.partial_rollout
    native_train.create_training_models=partial(create_training_models,actor_cls=TrackingMegatronTrainRayActor)
    (run/'resolved-args.json').write_text(json.dumps(vars(args),indent=2,default=str)+'\n')
    if os.environ.get('MM_VALIDATE_ONLY')=='1':return
    configure_logger()
    keys=['PYTHONPATH','MM_OPD_RUN','MM_OPD_SOURCE','WANDB_MODE','CUDA_DEVICE_MAX_CONNECTIONS',
          'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
          'OMP_NUM_THREADS','TOKENIZERS_PARALLELISM']
    keys += [k for k in os.environ if k.startswith(('NCCL_','GLOO_','NVSHMEM_'))]
    ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{k:os.environ[k] for k in keys if k in os.environ}})
    try:
        native_train.main(args,restore)
        (run/'training-finished.json').write_text('{"status":"passed"}\n')
    finally:
        if os.environ.get('MOSAIC_ONLINE')=='1':
            from turn_stream_contract import save
            save(run/'turn-stream/stop.json',{'reason':'trainer_exit'})
        ray.shutdown()
if __name__=='__main__':main()
