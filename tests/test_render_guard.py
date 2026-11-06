"""CPU-only fault injection into private fixtures, never the kernel log."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("MOSAIC_RENDER_HOST", "fixture-render-host")
from mosaic_distill.render_guard import RENDER_HOST, XID109, new_kernel_records, require_render_guard

XID = '<3>[ 12345.670000] NVRM: Xid (PCI:0000:19:00): 109, pid=12, name=thor-CloudRende, errorString CTX SWITCH TIMEOUT'
RUNNER = '''
import json, pathlib, socket, sys
from mosaic_distill.render_guard import supervise, RENDER_HOST
root=pathlib.Path(sys.argv[1]); socket.gethostname=lambda: RENDER_HOST
def read():
    value=(root/'kernel').read_text()
    if value=='DENIED': raise PermissionError('fixture: kernel read unavailable')
    return value.splitlines()
raise SystemExit(supervise([sys.executable, str(root/'work.py'), str(root)], root/sys.argv[2], root/'state', read_kernel=read))
'''
WORKER = '''
import os,pathlib,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1])
(root/'started').write_text(str(os.getpid()))
if (root/'short').exists():sys.exit(0)
if (root/'trip').exists():
    import socket
    from mosaic_distill.render_guard import fail_render,RENDER_HOST
    socket.gethostname=lambda:RENDER_HOST
    fail_render('fixture black frame')
child=subprocess.Popen([sys.executable,'-c', "import os,pathlib,signal,sys,time;os.setsid();signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)",str(root/'detached-pid')])
try:
    while True:
        (root/'heartbeat').write_text(str(time.monotonic()));time.sleep(.03)
finally:
    (root/'unsafe-finally-restart').write_text('must never execute on fault')
'''


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'runner.py').write_text(RUNNER)
        (self.root/'work.py').write_text(WORKER)
        self.set_log('<6>[ 1.000000] Boot ready\n')
        self.jobs=[]
        self.addCleanup(self.cleanup)
    def cleanup(self):
        for p in self.jobs:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
    def set_log(self,text):
        p=self.root/'kernel.tmp';p.write_text(text);p.replace(self.root/'kernel')
    def start(self,folder='run'):
        log=(self.root/(folder+'.log')).open('w');self.addCleanup(log.close)
        p=subprocess.Popen([sys.executable,str(self.root/'runner.py'),str(self.root),folder],stdout=log,stderr=log)
        self.jobs.append(p);return p
    def wait_for(self,name):
        end=time.monotonic()+8
        while not (self.root/name).exists():
            if time.monotonic()>end:self.fail('timeout waiting for '+name)
            time.sleep(.03)
    def assert_stopped(self,p):
        self.assertEqual(p.wait(timeout=15),109)
        self.assertTrue((self.root/'state'/RENDER_HOST/'BLOCKED.json').exists())
        self.assertFalse((self.root/'unsafe-finally-restart').exists())
        detached=self.root/'detached-pid'
        if detached.exists():
            pid=int(detached.read_text());stat=Path(f'/proc/{pid}/stat')
            self.assertTrue(not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z')
        proof=json.loads((self.root/'run/render-guard/failure.json').read_text())
        self.assertEqual(proof['cleanup']['survivors'],[])
    def test_xid_kills_detached_children_and_blocks_restart(self):
        p=self.start();self.wait_for('detached-pid')
        start=time.monotonic();self.set_log('<6>[ 1.000000] Boot ready\n'+XID+'\n')
        self.assert_stopped(p)
        self.assertLess(time.monotonic()-start,3)
        restart=self.start('retry');self.assertNotEqual(restart.wait(timeout=8),0)
        self.assertFalse((self.root/'retry/render-guard/started.json').exists())
    def test_permission_loss_stops_running_queue(self):
        p=self.start();self.wait_for('detached-pid');self.set_log('DENIED');self.assert_stopped(p)
    def test_cursor_loss_stops_running_queue(self):
        p=self.start();self.wait_for('detached-pid');self.set_log('<6>[ 2.000000] log cleared\n');self.assert_stopped(p)
    def test_stop_request_blocks_parent_cleanup_restart(self):
        (self.root/'trip').touch();p=self.start();self.assert_stopped(p)
    def test_existing_xid_prevents_any_worker_start(self):
        self.set_log(XID+'\n');p=self.start();self.assert_stopped(p)
        self.assertFalse((self.root/'started').exists())
    def test_unreadable_baseline_prevents_any_worker_start(self):
        self.set_log('DENIED');p=self.start();self.assert_stopped(p)
        self.assertFalse((self.root/'started').exists())
    def test_success_and_other_xid_do_not_false_trip(self):
        self.set_log(XID.replace(': 109,',': 43,')+'\n');(self.root/'short').touch()
        p=self.start();self.assertEqual(p.wait(timeout=8),0)
        self.assertTrue((self.root/'run/render-guard/completion.json').exists())
        self.assertFalse((self.root/'state'/RENDER_HOST/'BLOCKED.json').exists())
    def test_sigterm_stops_descendants_without_finally(self):
        p=self.start();self.wait_for('detached-pid');p.terminate();self.assert_stopped(p)
    def test_lossless_ring_rotation_and_exact_error_code(self):
        old=['a','b'];self.assertEqual(new_kernel_records(old,['a','b','c']),['c'])
        old=[str(i) for i in range(40)]
        self.assertEqual(new_kernel_records(old,old[10:]+['new']),['new'])
        self.assertTrue(XID109.search(XID));self.assertFalse(XID109.search(XID.replace(': 109,',': 1090,')))
    def test_wrong_host_or_missing_supervisor_refuses_rendering(self):
        with patch('socket.gethostname',return_value='gpu0909'):
            with self.assertRaisesRegex(RuntimeError,'explicitly configured render host'):require_render_guard()
        with patch('socket.gethostname',return_value=RENDER_HOST),patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'entire queue'):require_render_guard()


if __name__=='__main__':unittest.main()
