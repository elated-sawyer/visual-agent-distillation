"""Frozen teacher responses on the precommitted OPD prefix draw order."""
import asyncio, copy, hashlib, json, math, os, sys, time
from pathlib import Path
import aiohttp
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from slime.utils.data import Dataset
from slime.rollout.sglang_rollout import _prepare_prompt_ids
from slime.utils.processing_utils import encode_image_for_rollout_engine

run=Path(sys.argv[1]);cfg=json.loads((run/'config.json').read_text())
source=Path(cfg['opd_run'])
rows=[json.loads(s) for s in (source/'prefixes.jsonl').read_text().splitlines()]
assert len(rows)==cfg['prefix_draws']
tok=AutoTokenizer.from_pretrained(cfg['teacher']);proc=AutoProcessor.from_pretrained(cfg['teacher'])
image_pad=AutoConfig.from_pretrained(cfg['teacher']).image_token_id
data=Dataset(str(source/'pool.jsonl'),tok,proc,None,prompt_key='messages',apply_chat_template=True,
             apply_chat_template_kwargs={'enable_thinking':True})
expected=json.loads((source/'input-audit.json').read_text())['samples']
payloads=[];prefix_ids=[]
for index in range(len(data)):
    sample=copy.deepcopy(data[index]);ids=list(_prepare_prompt_ids(sample,tok,proc))
    assert len(ids)==expected[index]['prompt_tokens']
    assert len(ids)+cfg['response_limit']<=cfg['context_limit']
    assert sample.metadata['task']['split']=='train'
    payloads.append({'text':sample.prompt,'image_data':[encode_image_for_rollout_engine(x) for x in sample.multimodal_inputs['images']]})
    prefix_ids.append(ids)
assert len(list((run/'labels').glob('*.json')))==cfg['reused_count']
(run/'prefix-token-ids.json').write_text(json.dumps(prefix_ids))
completed={i:json.loads((run/f'labels/{i:04d}.json').read_text())['response_tokens'] for i in range(cfg['reused_count'])};start=time.time()
for name,digest in json.loads((run/'reused-labels-sha256.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
def progress():
    out={'completed':len(completed),'total':len(rows),'response_tokens':sum(completed.values()),'time':time.time(),'elapsed_seconds':time.time()-start}
    tmp=run/'progress.tmp';tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(run/'progress.json')

async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600),connector=aiohttp.TCPConnector(limit=40)) as session:
        async def request(i,port):
            row=rows[i];pool=row['metadata']['pool_index'];md=row['metadata']
            assert md['draw_index']==i
            payload=dict(payloads[pool])
            payload.update(return_logprob=True,logprob_start_len=0,
              sampling_params={'temperature':1,'top_p':1,'top_k':-1,'max_new_tokens':cfg['response_limit'],
                'skip_special_tokens':True,'no_stop_trim':True,'spaces_between_special_tokens':False,
                'sampling_seed':cfg['label_seed']+i})
            before=time.time()
            async with session.post(port.rstrip('/')+'/generate',json=payload) as response:
                response.raise_for_status();answer=await response.json()
            meta=answer['meta_info'];ids=[int(x[1]) for x in meta['output_token_logprobs']]
            assert 0<len(ids)<=cfg['response_limit']
            assert all(math.isfinite(x[0]) for x in meta['output_token_logprobs'])
            returned=meta['input_token_logprobs'];wanted=prefix_ids[pool]
            assert len(returned)==len(wanted),(i,len(returned),len(wanted))
            # Image-pad IDs are replaced by engine cache identities; text IDs must be exact.
            bad=[j for j,(a,b) in enumerate(zip(wanted,returned,strict=True)) if a!=image_pad and a!=b[1]]
            if bad:
                (run/f'failed-response-{i}.json').write_text(json.dumps({'expected_ids':wanted,'output':answer}))
            assert not bad,{'draw':i,'wrong_text_positions':bad[:10]}
            assert meta['finish_reason']['type'] in ('stop','length')
            record={'draw_index':i,'pool_index':pool,'prefix_sha256':md['prefix_sha256'],
                'metadata':md,'response_ids':ids,'response_text':answer['text'],
                'response_tokens':len(ids),'prompt_tokens':len(wanted),
                'finish_reason':meta['finish_reason'],'sampling_seed':cfg['label_seed']+i,
                'teacher':cfg['teacher'],'all_prefix_text_ids_verified':True,
                'elapsed_seconds':time.time()-before,'service_port':port}
            (run/'labels'/f'{i:04d}.json').write_text(json.dumps(record,ensure_ascii=False)+'\n')
            completed[i]=len(ids);progress()
            print(f'label {i}: {len(ids)} tokens; {len(completed)}/{len(rows)}',flush=True)
        async def service(port,indices):
            if not indices: return
            await request(indices[0],port)
            sem=asyncio.Semaphore(cfg['concurrency_per_service'])
            async def bounded(i):
                async with sem:await request(i,port)
            await asyncio.gather(*(bounded(i) for i in indices[1:]))
        await asyncio.gather(*(service(port,list(range(cfg['reused_count']+k,len(rows),len(cfg['teacher_urls'])))) for k,port in enumerate(cfg['teacher_urls'])))
    assert set(completed)==set(range(len(rows)))
    manifest={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run/'labels').glob('*.json'))}
    (run/'labels-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (run/'audit.json').write_text(json.dumps({'status':'passed','labels':len(rows),'reused_labels':cfg['reused_count'],'new_labels':len(rows)-cfg['reused_count'],'response_tokens':sum(completed.values()),
       'prefixes_identical_to_opd_draws':True,'outcome_filter':False,'environment_interactions':0,
       'native_thinking':True,'elapsed_seconds':time.time()-start},indent=2)+'\n')

asyncio.run(main())
