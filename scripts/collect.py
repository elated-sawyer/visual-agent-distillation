"""Collect or evaluate a frozen panel using the original model responses."""
import argparse,asyncio,copy,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mosaic_distill.assets import ImageStore
from mosaic_distill.records import to_chat_messages
from mosaic_distill.rollout import collect_episode
from mosaic_distill.limits import ContextBudgetExhausted,remaining_from_context_error

def run_episode(job):
    import requests
    index,item,protocol,endpoint,model,output,context,role=job
    folder=Path(output)/f'{index:05d}';folder.mkdir(parents=True,exist_ok=False)
    store=ImageStore(Path(output)/'assets');cfg=copy.deepcopy(protocol);calls=[]
    if cfg['schema']=='navigation-protocol-v1':
        from mosaic_distill.navigation import VagenNavigation
        cfg['runtime']['root']=str((ROOT/cfg['runtime']['root']).resolve())
        env=VagenNavigation(cfg,item['split'],store)
    elif cfg['schema']=='spatial-protocol-v1':
        from mosaic_distill.clarified_spatial import VagenSpatialGym
        cfg['env_config']['data_dir']=str((ROOT/cfg['env_config']['data_dir']).resolve())
        env=VagenSpatialGym(cfg,item['split'],store)
    else:
        from mosaic_distill.sokoban import VagenSokoban
        env=VagenSokoban(cfg,item['split'],store)
    session=requests.Session();session.trust_env=False
    async def policy(messages):
        payload={'model':model,'messages':to_chat_messages(messages,store),'max_tokens':32768,
                 'temperature':1.0,'top_p':1.0,'top_k':-1,'seed':item['sampling_seed']+len(calls),
                 'separate_reasoning':False,'chat_template_kwargs':{'enable_thinking':True},
                 'skip_special_tokens':True,'stream':False}
        for attempt in range(2):
            reply=session.post(endpoint.rstrip('/')+'/v1/chat/completions',json=payload,timeout=3600)
            if reply.ok:break
            try:error=reply.json()
            except ValueError:error={}
            remaining=remaining_from_context_error(reply.status_code,error,context,payload['max_tokens'])
            if remaining is None or attempt:reply.raise_for_status();raise RuntimeError('Unexpected model service response')
            if remaining==0:raise ContextBudgetExhausted()
            payload['max_tokens']=remaining
        data=reply.json();choice=data['choices'][0];message=choice['message']
        if message.get('reasoning_content'):raise ValueError('Service split thinking from original content; configure separate_reasoning=false')
        text=message['content']
        if not isinstance(text,str) or choice['finish_reason'] not in ['stop','length']:raise ValueError('Invalid model completion')
        usage=data['usage']
        if usage['prompt_tokens']+usage['completion_tokens']>context:raise ValueError('Service exceeded context budget')
        calls.append({'sampling_seed':payload['seed'],'usage':usage,'finish_reason':choice['finish_reason']})
        return text
    try:
        ep=asyncio.run(collect_episode(env,policy,item['seed'],{'kind':role,'model':model}))
        if ep['status']!='complete':raise RuntimeError('Infrastructure failures are unscored')
        if 'task_id' in item and ep['task']['task_id']!=item['task_id']:raise ValueError('Task identity drift')
        row={'task_id':ep['task']['task_id'],'repeat':item['repeat'],'split':item['split'],'success':int(ep['success'])}
        if cfg['schema']=='spatial-protocol-v1':
            from mosaic_distill.spatial import episode_metrics
            row.update(episode_metrics(ep))
        (folder/'episode.json').write_text(json.dumps(ep,ensure_ascii=False)+'\n')
        (folder/'calls.json').write_text(json.dumps(calls)+'\n')
        return row
    finally:session.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--endpoints',nargs='+',required=True);p.add_argument('--served-model',default='mm-student')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--context',type=int,required=True);p.add_argument('--role',choices=['student','teacher'],default='student')
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());panel=json.loads(a.panel.read_text())
    if not panel or a.workers<1:p.error('Nonempty panel and positive worker count required')
    ids=[(x['seed'],x['repeat'],x['split']) for x in panel]
    if len(set(ids))!=len(ids):p.error('Duplicate evaluation slots')
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    jobs=[(i,x,protocol,a.endpoints[i%len(a.endpoints)],a.served_model,str(a.output.resolve()),a.context,a.role) for i,x in enumerate(panel)]
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(run_episode,j) for j in jobs]):
                records.append(future.result())
                (a.output/'progress.json').write_text(json.dumps({'completed':len(records),'total':len(panel)})+'\n')
        records.sort(key=lambda x:(x['task_id'],x['repeat']))
        (a.output/'results.json').write_text(json.dumps({'status':'passed','records':records},indent=2)+'\n')
    except BaseException as exc:
        (a.output/'failed.json').write_text(json.dumps({'type':type(exc).__name__,'completed':len(records)})+'\n')
        raise
if __name__=='__main__':main()
