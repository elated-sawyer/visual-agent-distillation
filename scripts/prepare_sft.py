"""Create teacher-label requests, or attach audited labels to a matched SFT run."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['request','attach']);p.add_argument('--opd-run',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--teacher',type=Path)
    p.add_argument('--teacher-urls',nargs='+');p.add_argument('--output',type=Path)
    p.add_argument('--concurrency-per-service',type=int,default=8)
    a=p.parse_args();source=a.opd_run.resolve();labels=a.labels.resolve()
    cfg=json.loads((source/'config.json').read_text())
    if a.action=='request':
        if not a.teacher or not a.teacher_urls:p.error('request requires --teacher and --teacher-urls')
        if a.concurrency_per_service<1:p.error('concurrency must be positive')
        labels.mkdir(parents=True,exist_ok=False);(labels/'labels').mkdir()
        config={**cfg,'opd_run':str(source),'teacher':str(a.teacher.resolve()),'teacher_urls':a.teacher_urls,
                'prefix_draws':cfg['updates']*cfg['batch_size'],'reused_count':0,'label_seed':20261022,
                'concurrency_per_service':a.concurrency_per_service}
        (labels/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (labels/'reused-labels-sha256.json').write_text('{}\n')
    else:
        if not a.output:p.error('attach requires --output')
        audit=json.loads((labels/'audit.json').read_text());assert audit['status']=='passed'
        rows=[json.loads(line) for line in (source/'prefixes.jsonl').read_text().splitlines()]
        assert audit['labels']==len(rows)
        sealed=json.loads((labels/'labels-sha256.json').read_text())
        for i,row in enumerate(rows):
            md=row['metadata'];path=labels/f'labels/{i:04d}.json';raw=path.read_bytes();lab=json.loads(raw)
            digest=hashlib.sha256(raw).hexdigest();assert digest==sealed[f'labels/{i:04d}.json']
            assert lab['draw_index']==i and lab['prefix_sha256']==md['prefix_sha256'] and lab['pool_index']==md['pool_index']
            md.update(sft_order=i,label_sha256=digest)
        out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(source/'pool.jsonl',out/'pool.jsonl')
        (out/'prefixes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        cfg.update(labels_run=str(labels));(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        print('Created matched SFT inputs. Run training/validate_inputs.py before training.')
if __name__=='__main__':main()
