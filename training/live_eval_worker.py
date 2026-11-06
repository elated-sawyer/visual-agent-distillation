"""Execute one requested frozen panel on a separately deployed environment host."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from evaluation_contract import validate_panel

def evaluate(path):
    req=json.loads(path.read_text());panel=req['panel'];folder=Path(req['evaluation'])
    manifest=folder/'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=req['eval_source_sha256']:raise ValueError('Panel manifest changed')
    spec=json.loads(manifest.read_text());cfg=json.loads((Path(req['run'])/'config.json').read_text())
    output=folder/'collection'
    cmd=[sys.executable,str(ROOT/'scripts/collect.py'),'--protocol',spec['protocol'],'--panel',spec['panel'],
         '--endpoints',*req['engines'],'--output',str(output),'--context',str(cfg['evaluation_context_limit'])]
    subprocess.run(cmd,check=True)
    records=json.loads((output/'results.json').read_text())['records'];validate_panel(panel,records)
    result={'status':'passed','request_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'step':req['step'],'expected_version':req['expected_version'],'records':records}
    (path.parent/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request',type=Path);a=p.parse_args()
    try:evaluate(a.request)
    except BaseException as e:
        (a.request.parent/'failed.json').write_text(json.dumps({'type':type(e).__name__})+'\n');raise
