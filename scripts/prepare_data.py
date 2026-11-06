"""Build replay inputs from an explicit, already selected trajectory manifest."""
import argparse, copy, hashlib, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mosaic_distill.records import export_prefixes, fingerprint
from mosaic_distill.assets import ImageStore

def prepare(args):
    cfg=json.loads(args.recipe.read_text())
    cfg.update(student=str(args.student.resolve()),trajectory_count=0,training_indices=[])
    if args.updates is not None:cfg['updates']=args.updates
    if args.batch_size is not None:cfg['batch_size']=args.batch_size
    if cfg['updates']<=0 or cfg['batch_size']<=0:raise ValueError('Positive update/batch budgets required')
    pool=[]
    if args.demo:
        groups={}
        for p in sorted((ROOT/'data/prefixes').glob('*.json')):
            row=json.loads(p.read_text());cid=row['candidate_id']
            index=groups.setdefault(cid,len(groups));messages=copy.deepcopy(row['messages'])
            for m in messages:
                for part in m['content']:
                    if part['type']=='image':
                        image=(ROOT/part['src']).resolve()
                        if not image.is_relative_to(ROOT) or not image.is_file():raise ValueError('Invalid example image')
                        part.clear();part.update(type='image',image=str(image))
            pool.append({'messages':messages,'metadata':{'episode_sha256':hashlib.sha256((ROOT/f'data/trajectories/{cid}.json').read_bytes()).hexdigest(),
                'prefix_sha256':fingerprint(row),'turn_index':row['turn_index'],'collection_index':index,
                'source_role':'student','task':{'split':'train','task_id':'demo:'+cid}}})
    else:
        manifest=json.loads(args.episodes.read_text())
        if not isinstance(manifest,list) or not manifest:raise ValueError('Expected a nonempty list of selected episode paths')
        store=ImageStore(args.assets)
        for index,name in enumerate(manifest):
            ep=json.loads((args.episodes.parent/name).read_text())
            for prefix in export_prefixes(ep,image_requirement='history'):
                messages=copy.deepcopy(prefix['messages'])
                for m in messages:
                    for part in m['content']:
                        if part['type']=='image_ref':
                            ref=part['image'];store.read(ref);path=store._path(ref).resolve()
                            part.clear();part.update(type='image',image=str(path))
                md={k:prefix[k] for k in ['episode_sha256','prefix_sha256','turn_index','task']}
                md.update(collection_index=index,source_role=ep['source']['kind'])
                pool.append({'messages':messages,'metadata':md})
    groups={}
    for i,row in enumerate(pool):groups.setdefault(row['metadata']['episode_sha256'],[]).append(i)
    if not groups:raise ValueError('No valid training prefixes')
    if any(r['metadata']['source_role']!='student' for r in pool):
        raise ValueError('Published recipe uses student-collected prefixes; use teacher generation for SFT responses')
    cfg.update(trajectory_count=len(groups),training_indices=sorted({r['metadata']['collection_index'] for r in pool}))
    rng=random.Random(cfg['seed']);draws=[]
    for i in range(cfg['updates']*cfg['batch_size']):
        j=rng.choice(rng.choice(list(groups.values())))
        row=copy.deepcopy(pool[j]);row['metadata'].update(draw_index=i,pool_index=j);draws.append(row)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name,rows in [('pool.jsonl',pool),('prefixes.jsonl',draws)]:
        with (out/name).open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'run':str(out),'trajectories':len(groups),'prefixes':len(pool),'draws':len(draws)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo',action='store_true');group.add_argument('--episodes',type=Path)
    p.add_argument('--assets',type=Path,default=ROOT/'outputs/assets')
    p.add_argument('--recipe',type=Path,default=ROOT/'configs/sokoban.json')
    p.add_argument('--student',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int);p.add_argument('--batch-size',type=int)
    prepare(p.parse_args())
