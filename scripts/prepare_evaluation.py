"""Attach explicit monitoring/final panels for native live-weight evaluation."""
import argparse,copy,hashlib,json,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'training')]
from mosaic_distill.records import fingerprint
from evaluation_contract import validate_schedule

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--panels',type=Path,required=True,help='List of name/scope/environment/protocol/panel descriptors')
    p.add_argument('--inference-ip',required=True);p.add_argument('--rollout-gpus',type=int,default=4)
    a=p.parse_args();run=a.run.resolve();cfg=json.loads((run/'config.json').read_text())
    specs=json.loads(a.panels.read_text());catalog=[]
    if any(s['scope'] not in ['monitor','final'] for s in specs):p.error('Expected monitor/final scope')
    for s in specs:
        envpath=(a.panels.parent/s['protocol']).resolve();panelpath=(a.panels.parent/s['panel']).resolve()
        rows=json.loads(panelpath.read_text());protocol=json.loads(envpath.read_text())
        if any('task_id' not in r for r in rows):p.error('Live panels must declare frozen task_id for every slot')
        catalog.append({**s,'protocol':str(envpath),'panel':str(panelpath),
                        'episodes':len(rows),'slots':[[r['task_id'],r['repeat']] for r in rows]})
    evaluations=[]
    for step in range(cfg['evaluation_interval'],cfg['updates']+1,cfg['evaluation_interval']):
        panels=[]
        for s in catalog:
            if s['scope']=='final' and step!=cfg['updates']:continue
            target=run/'evaluation-panels'/f"{step:04d}-{s['name']}";target.mkdir(parents=True,exist_ok=False)
            (target/'manifest.json').write_text(json.dumps(s,indent=2)+'\n')
            panels.append({k:s[k] for k in ['name','scope','environment','episodes','slots']}|{'eval':str(target)})
        evaluations.append({'step':step,'panels':panels,'episodes':sum(x['episodes'] for x in panels)})
    cfg.update(curve_evaluations=evaluations,curve_smoke=False,curve_environment=cfg['environment'],
               inference_ip=a.inference_ip,rollout_gpus=a.rollout_gpus,render_queue=str(run/'evaluation-worker'))
    validate_schedule(cfg);(run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    first=(run/'pool.jsonl').read_text().splitlines()[0]
    (run/'eval-dataset.jsonl').write_text(first+'\n')
    print('Prepared live evaluations. Start scripts/evaluation_worker.py before training.')
if __name__=='__main__':main()
