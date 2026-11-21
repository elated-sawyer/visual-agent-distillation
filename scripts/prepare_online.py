"""Connect a training-task catalogue to the 16-lane online adapter."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'training')]
from online_contract import validate_catalogue
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--catalogue',type=Path,required=True);p.add_argument('--render-hostname',required=True)
    a=p.parse_args();run=a.run.resolve();c=json.loads((run/'config.json').read_text())
    if c['batch_size']!=64:p.error('Validated online adapter requires batch 64')
    cat=json.loads(a.catalogue.read_text());validate_catalogue(cat)
    c.update(online_render_hostname=a.render_hostname,online_request_timeout_seconds=3600,render_rollout_gpus=4,
             online_environment_rollouts=True)
    (run/'online-catalogue.json').write_text(json.dumps(cat,indent=2)+'\n')
    (run/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (run/'source').symlink_to(ROOT/'training',target_is_directory=True)
    (run/'turn-stream').mkdir()
