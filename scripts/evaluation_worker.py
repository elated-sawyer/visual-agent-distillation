"""Service offline OPD/SFT live evaluations; online lanes use their own supervisor."""
import argparse,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    while not (a.run/'training-finished.json').exists():
        for request in sorted((a.run/'live-evaluations').glob('*/request.json')):
            if (request.parent/'result.json').exists():continue
            if (request.parent/'failed.json').exists():raise RuntimeError('Evaluation previously failed; inspect before restarting')
            subprocess.run([sys.executable,str(ROOT/'training/live_eval_worker.py'),str(request)],check=True)
        time.sleep(2)
