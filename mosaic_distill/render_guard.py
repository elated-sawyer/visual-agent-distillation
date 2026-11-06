"""Fail-closed, node-scoped supervision for the explicitly configured render host rendering queues.

No GPU/framework imports. Run the complete queue as a child of this supervisor;
detached descendants are still owned through Linux's child-subreaper facility.
"""
from __future__ import annotations

import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

RENDER_HOST = os.environ.get('MOSAIC_RENDER_HOST')
LEASE_ENV = 'MOSAIC_RENDER_GUARD'
XID109 = re.compile(r'\bNVRM:\s*Xid\s*\([^\n]*\):\s*109\s*,')
POLL_SECONDS = 0.5
LEASE_SECONDS = 10


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]), 'start': fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def kernel_snapshot():
    # /dev/kmsg is not exposed in this container. klogctl/dmesg reads still work.
    r = subprocess.run(['dmesg', '--raw'], capture_output=True, text=True, timeout=2)
    if r.returncode != 0 or r.stderr.strip():
        raise RuntimeError(f'Kernel monitor exit {r.returncode}: ' + r.stderr[-1000:])
    lines = r.stdout.splitlines()
    if not lines:
        raise RuntimeError('Empty kernel log; cannot establish a monitoring cursor')
    return lines


def new_kernel_records(previous, current):
    if current[:len(previous)] == previous:
        return current[len(previous):]
    # Account for ring-buffer rotation without silently losing the cursor.
    anchor = previous[-min(20, len(previous)):]
    positions = [i for i in range(len(current) - len(anchor) + 1)
                 if current[i:i + len(anchor)] == anchor]
    if len(positions) != 1:
        raise RuntimeError('Kernel log cursor lost or ambiguous; rendering must stop')
    return current[positions[0] + len(anchor):]


def require_render_guard():
    if socket.gethostname() != RENDER_HOST:
        raise RuntimeError('Rendering is authorized only on the explicitly configured render host')
    path = os.environ.get(LEASE_ENV)
    if not path:
        raise RuntimeError('Rendering requires run_with_render_guard.py around the entire queue')
    lease = json.loads(Path(path).read_text())
    if lease['host'] != RENDER_HOST or lease['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Render guard host/boot identity mismatch')
    if Path(lease['latch']).exists() or (Path(path).parent/'failure-request.json').exists():
        raise RuntimeError('Render guard is tripped; automatic restart is forbidden')
    if lease['status'] != 'running' or time.monotonic() - lease['heartbeat_monotonic'] > LEASE_SECONDS:
        raise RuntimeError('Render guard heartbeat is stale')
    identity = process_identity(lease['pid'])
    if identity is None or identity['start'] != lease['start'] or identity['state'] in {'Z', 'T', 't'}:
        raise RuntimeError('Render guard supervisor is no longer active')
    pid = os.getpid()
    while pid > 1 and pid != lease['pid']:
        proc = process_identity(pid)
        if proc is None:
            break
        pid = proc['ppid']
    if pid != lease['pid']:
        raise RuntimeError('Renderer is outside the supervised process tree')
    return Path(path), lease


def fail_render(reason):
    path, lease = require_render_guard()
    failure = {'reason': reason, 'host': RENDER_HOST, 'pid': os.getpid(), 'time': time.time()}
    # A persistent node latch prevents a caught exception from starting a new renderer.
    write_json(lease['latch'], failure)
    write_json(path.parent/'failure-request.json', failure)
    raise RuntimeError(reason)


def stop_descendants():
    """Freeze parents before children; do not run their automatic-restart finally blocks."""
    frozen = {}
    for _ in range(50):
        rows = {}
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                row = process_identity(int(p.name))
                if row is not None and row['state'] != 'Z':
                    rows[row['pid']] = row
        parents = {os.getpid()}
        discovered = []
        while parents:
            children = [r for r in rows.values() if r['ppid'] in parents]
            for row in children:
                pid = row['pid']
                current = process_identity(pid)
                if current and current['start'] == row['start']:
                    try:
                        os.kill(pid, signal.SIGSTOP)
                        frozen[pid] = row['start']
                        discovered.append(pid)
                    except ProcessLookupError:
                        pass
            parents = {r['pid'] for r in children}
        if all(p in frozen for p in discovered):
            # Re-scan once after parents were frozen to catch a fork during the first scan.
            if not discovered or _ > 0:
                break
    for pid, start in frozen.items():
        current = process_identity(pid)
        if current and current['start'] == start and current['state'] != 'Z':
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    survivors = []
    while time.monotonic() < deadline:
        survivors = [pid for pid, start in frozen.items()
                     if (r := process_identity(pid)) and r['start'] == start and r['state'] != 'Z']
        if not survivors:
            break
        time.sleep(0.05)
    return {'terminated_pids': sorted(frozen), 'survivors': survivors}


def supervise(command, run_dir, state_root, *, read_kernel=kernel_snapshot):
    if socket.gethostname() != RENDER_HOST:
        raise RuntimeError('Rendering is authorized only on the explicitly configured render host')
    if not command:
        raise ValueError('A queue command is required')
    run_dir = Path(run_dir).resolve()
    state = Path(state_root).resolve()/RENDER_HOST
    state.mkdir(parents=True, exist_ok=True)
    latch = state/'BLOCKED.json'
    with (state/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if latch.exists():
            raise RuntimeError(f'Node rendering is locked: {latch}; no automatic clearing')
        guard_dir = run_dir/'render-guard'
        guard_dir.mkdir(parents=True, exist_ok=False)
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        identity = process_identity(os.getpid())
        lease = {**identity, 'host': RENDER_HOST, 'boot_id': boot_id, 'latch': str(latch),
                 'status': 'starting', 'heartbeat_monotonic': time.monotonic()}
        lease_path = guard_dir/'lease.json'
        child = None
        previous_handlers = {}
        def interrupted(sig, frame):
            raise InterruptedError(f'Supervisor received signal {sig}')
        try:
            # Descendants with setsid()/double fork remain reclaimable by this supervisor.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                raise OSError(ctypes.get_errno(), 'Cannot establish subreaper')
            previous = read_kernel()
            (guard_dir/'kernel-baseline.log').write_text('\n'.join(previous) + '\n')
            # An earlier Xid109 in the current readable boot log requires human investigation.
            old_errors = [line for line in previous if XID109.search(line)]
            if old_errors:
                raise RuntimeError('Existing Xid109 in node kernel log: ' + old_errors[-1])
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, interrupted)
            lease.update(status='running', heartbeat_monotonic=time.monotonic())
            write_json(lease_path, lease)
            child = subprocess.Popen(command, env=os.environ | {LEASE_ENV: str(lease_path)},
                                     start_new_session=True)
            write_json(guard_dir/'started.json', {'pid': child.pid, 'time': time.time(),
                                                'supervisor': identity, 'host': RENDER_HOST})
            while True:
                if (guard_dir/'failure-request.json').exists() or latch.exists():
                    raise RuntimeError('Renderer requested stop; see failure-request.json/node latch')
                current = read_kernel()
                new = new_kernel_records(previous, current)
                previous = current
                errors = [line for line in new if XID109.search(line)]
                if errors:
                    (guard_dir/'xid109.log').write_text('\n'.join(errors) + '\n')
                    raise RuntimeError('New Xid109: ' + errors[0])
                lease['heartbeat_monotonic'] = time.monotonic()
                write_json(lease_path, lease)
                if child.poll() is not None:
                    if child.returncode != 0:
                        raise RuntimeError(f'Supervised queue failed, returncode={child.returncode}')
                    break
                time.sleep(POLL_SECONDS)
            cleanup = stop_descendants()
            if cleanup['survivors']:
                raise RuntimeError(f'Child processes did not stop: {cleanup}')
            lease['status'] = 'complete'
            write_json(lease_path, lease)
            write_json(guard_dir/'completion.json', {'status': 'passed', 'time': time.time(), **cleanup})
            return 0
        except BaseException as exc:
            failure = {'status': 'blocked', 'host': RENDER_HOST, 'time': time.time(),
                       'boot_id': boot_id, 'reason': f'{type(exc).__name__}: {exc}',
                       'run_dir': str(run_dir), 'automatic_restart': False}
            # Cleanup must still happen if shared storage becomes unavailable.
            try:
                write_json(latch, failure)
                write_json(guard_dir/'failure.json', failure)
            finally:
                cleanup = stop_descendants()
                if child is not None:
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            failure['cleanup'] = cleanup
            write_json(guard_dir/'failure.json', failure)
            lease.update(status='blocked', heartbeat_monotonic=time.monotonic())
            write_json(lease_path, lease)
            return 109
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
