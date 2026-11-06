"""Persistent renderer lanes, supervised by the unchanged canonical Xid guard."""
import asyncio,copy,os,socket,sys,time
from pathlib import Path
# Script directory precedes PYTHONPATH; select the complete frozen environment package.

from online_contract import read,digest
from turn_stream_contract import task_for_lane,decision_dir,validate_response,save
from mosaic_distill.assets import ImageStore
from mosaic_distill.records import decision_messages
from mosaic_distill.guard_readiness import require_render_guard

class StreamStopped(BaseException):
    pass

async def lane_main(run,lane):
    cfg=read(run/'config.json');cat=read(run/'online-catalogue.json')
    root=run/'turn-stream';store=ImageStore(root/'assets')
    assert socket.gethostname()==cfg['online_render_hostname']
    sequence=0;cycle=0
    def check():
        require_render_guard()
        if (root/'failed.json').exists():
            raise RuntimeError('Stream parent failed or stopped without shutdown handshake')
        if (root/'stop.json').exists():raise StreamStopped()
    while True:
        check()
        task=task_for_lane(cat,cfg['seed'],lane,cycle)
        episode_id=cycle*16+lane
        if cat['environment']=='spatial':
            from mosaic_distill.clarified_spatial import VagenSpatialGym
            env=VagenSpatialGym(cat['protocol'],'train',store)
        elif cat['environment']=='navigation':
            from mosaic_distill.navigation import VagenNavigation
            env=VagenNavigation(cat['protocol'],'train',store,gpu_device=lane%cfg['render_rollout_gpus'])
        elif cat['environment']=='sokoban':
            from mosaic_distill.sokoban import VagenSokoban
            env=VagenSokoban(cat['protocol'],'train',store)
        else:raise ValueError('Unknown environment')
        episode={'schema':'mm-online-stream-episode-v1','status':'running','turns':[],
                 'source':{'kind':'student','model_path':cat['student'],'history_may_cross_updates':True},
                 'protocol_sha256':cat['protocol_sha256'],'online_episode_id':episode_id}
        versions=[]
        try:
            check();observation,actual=await env.reset(task['seed'])
            assert actual==task
            episode.update(task=actual,system=await env.system_prompt())
            for turn in range(cat['protocol']['max_turns']):
                check()
                episode['turns'].append({'observation':copy.deepcopy(observation)})
                item={'lane_id':lane,'sequence':sequence,'episode_id':episode_id,
                      'episode_cycle':cycle,'turn_index':turn,'task':task,
                      'protocol_sha256':cat['protocol_sha256'],'history_versions':list(versions),
                      'messages':decision_messages(episode,turn)}
                d=decision_dir(root,lane,sequence)
                assert not d.exists(),'Refuse to reuse an online decision'
                save(d/'request.json',item)
                save(root/'lanes'/f'{lane:02d}'/'ready.json',
                     {'lane_id':lane,'sequence':sequence,'request_sha256':digest(item),'time':time.time()})
                assigned_at=None
                while not (d/'response.json').exists():
                    check()
                    if (d/'claim.json').exists() and assigned_at is None:assigned_at=time.monotonic()
                    if assigned_at is not None and time.monotonic()-assigned_at>cfg['online_request_timeout_seconds']:
                        raise TimeoutError('Assigned policy response timeout')
                    await asyncio.sleep(.1)
                check()
                claim=read(d/'claim.json');response=read(d/'response.json')
                validate_response(item,claim,response)
                context_stop=response['status']=='context_limit'
                if context_stop:
                    episode['turns'].pop()
                    episode.update(status='complete',success=False,stop_reason='context_limit',
                                   final_observation=copy.deepcopy(observation))
                else:
                    text=response['response']
                    observation,reward,terminated,truncated,info=await env.step(text)
                    episode['turns'][-1].update(response=text,next_observation=copy.deepcopy(observation),
                        reward=reward,terminated=terminated,truncated=truncated,info=copy.deepcopy(info),
                        rollout_id=claim['rollout_id'],weight_version=claim['expected_version'])
                    versions.append(claim['expected_version'])
                    if terminated or truncated:
                        episode.update(status='complete',success=bool(info.get('success',False)),
                                       stop_reason='turn_limit' if truncated else 'terminated')
                completed=episode['status']=='complete'
                save(root/'episodes'/f'{episode_id:08d}.json',episode)
                save(d/'executed.json',{'status':'passed','lane_id':lane,'sequence':sequence,
                    'sample_index':claim['sample_index'],'expected_version':claim['expected_version'],
                    'request_sha256':digest(item),'response_sha256':digest(response),
                    'action_executed':not context_stop,'episode_completed':completed,
                    'episode_id':episode_id,'stop_reason':episode.get('stop_reason'),
                    'next_observation_sha256':digest(observation),'time':time.time()})
                sequence+=1
                if completed:break
            else:raise RuntimeError('Environment did not honor turn limit')
        finally:
            await env.close()
        cycle+=1

def main(run,lane):
    root=run/'turn-stream'
    try:asyncio.run(lane_main(run,lane))
    except StreamStopped:
        save(root/'lanes'/f'{lane:02d}'/'stopped.json',{'status':'passed','time':time.time()})
    except BaseException as exc:
        from mosaic_distill.error_diagnostics import exception_context
        save(root/'failed.json',{'lane_id':lane,'type':type(exc).__name__,
             'diagnostic':exception_context(exc,'persistent_environment_lane'),'time':time.time()})
        raise

if __name__=='__main__':
    main(Path(sys.argv[1]),int(sys.argv[2]))
