"""One canonical rendering lease covers persistent training lanes and live eval."""
import os,subprocess,sys,time
from pathlib import Path
from online_contract import read
from turn_stream_contract import save
from mosaic_distill.guard_readiness import require_render_guard

def main(run):
    root=run/'turn-stream';cfg=read(run/'config.json')
    children=[];logs=[];evaluation=None;eval_path=None
    try:
        require_render_guard()
        for lane in range(16):
            path=root/'lanes'/f'{lane:02d}';path.mkdir(parents=True,exist_ok=True)
            h=(path/'worker.log').open('w');logs.append(h)
            children.append(subprocess.Popen([sys.executable,'-B',str(run/'source/turn_stream_worker.py'),str(run),str(lane)],
                env=os.environ.copy(),stdout=h,stderr=subprocess.STDOUT))
        save(root/'supervisor-ready.json',{'pid':os.getpid(),'lanes':16,'time':time.time()})
        deadline=time.monotonic()+604800
        while not (root/'stop.json').exists():
            require_render_guard()
            assert time.monotonic()<deadline,'Stream lifetime exceeded'
            assert not (root/'failed.json').exists(),'Environment lane failed'
            assert all(p.poll() is None for p in children),'Persistent lane unexpectedly exited'
            if evaluation is not None:
                if evaluation.poll() is not None:
                    assert evaluation.returncode==0,'Live evaluation failed'
                    assert read(eval_path.parent/'result.json')['status']=='passed'
                    evaluation=None;eval_path=None
            else:
                for path in sorted((run/'live-evaluations').glob('*/request.json')):
                    if (path.parent/'result.json').exists():continue
                    # Learner only calls native eval after the round's all-lane quiescence barrier.
                    latest=read(root/'quiescent.json')
                    assert latest['status']=='passed'
                    assert not (root/'round-active.json').exists(),'Cannot evaluate during online actions'
                    h=(path.parent/'worker.log').open('w');logs.append(h)
                    evaluation=subprocess.Popen([sys.executable,'-B',str(run/'source/live_eval_worker.py'),str(path)],
                        env=os.environ.copy(),stdout=h,stderr=subprocess.STDOUT)
                    eval_path=path;break
            time.sleep(.2)
        if evaluation is not None:
            evaluation.wait(timeout=60)
            assert evaluation.returncode==0 and read(eval_path.parent/'result.json')['status']=='passed'
            evaluation=None
        for p in children:
            p.wait(timeout=60);assert p.returncode==0,'Lane failed to close'
        save(root/'completion.json',{'status':'passed','lanes_closed':16,'time':time.time()})
    except BaseException as exc:
        from mosaic_distill.error_diagnostics import exception_context
        save(root/'failed.json',{'type':type(exc).__name__,'diagnostic':exception_context(exc,'stream_supervisor'),'time':time.time()})
        raise
    finally:
        for p in children+([evaluation] if evaluation is not None else []):
            if p.poll() is None:p.terminate()
        for p in children+([evaluation] if evaluation is not None else []):
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in logs:h.close()

if __name__=='__main__':main(Path(sys.argv[1]))
