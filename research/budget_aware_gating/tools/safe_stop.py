#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

progress = Path('/data/dxl/results/budget_aware_gating/progress/master_progress.json')
state = json.loads(progress.read_text())
pid = state.get('launcher_pid')
if not pid:
    print('No recorded budget-aware launcher PID.')
    raise SystemExit(0)
proc = Path(f'/proc/{pid}')
try:
    uid = proc.stat().st_uid
    command = (proc / 'cmdline').read_bytes()
    environment = (proc / 'environ').read_bytes()
    cwd = Path(os.readlink(proc / 'cwd')).resolve()
    exe = Path(os.readlink(proc / 'exe')).resolve()
    pgid = os.getpgid(pid)
except FileNotFoundError:
    print('Recorded launcher is no longer running.')
    raise SystemExit(0)
allowed = (
    b'run_probe_smoke.py',
    b'run_screen_validation.py',
    b'run_budget_validation.py',
)
checks = {
    'pid': pid,
    'pgid': pgid,
    'user_uid': uid,
    'cwd': str(cwd),
    'exe': str(exe),
    'command_match': any(item in command for item in allowed),
    'environment_match': b'MATTERGEN_BUDGET_LAUNCHER=1' in environment,
    'cwd_match': cwd == Path('/data/dxl/mattergen_v1'),
    'uid_match': uid == os.getuid(),
}
print(json.dumps(checks, indent=2))
if not all(
    checks[key]
    for key in ('command_match', 'environment_match', 'cwd_match', 'uid_match')
):
    print('Safety verification failed; no signal sent.', file=sys.stderr)
    raise SystemExit(2)
Path('/data/dxl/results/budget_aware_gating/progress/stop_requested').write_text(
    'SIGINT requested\n'
)
os.killpg(pgid, signal.SIGINT)
print('SIGINT sent to verified project process group; no SIGKILL used.')
