# Resource audit

2026-09-14T12:17:48.464834+00:00

No processes terminated; no historical files/checkpoints/stashes modified.

## git status --short

```text
?? experiments/final_thesis_results/
?? experiments/innovation2_research_v2/
?? experiments/physics_sampling_candidates_summary.md

```

## git rev-parse HEAD

```text
1af301c5e534ff6b9b9f96ea5dab48036d9c46af

```

## git branch --show-current

```text
experiment/raab-sc-p0

```

## git stash list

```text
stash@{0}: On raab-sc-p0: preserve-raab-sc-p0
stash@{1}: On mattergen-rl-p0: preserve-mattergen-rl-p0
stash@{2}: On alm-gen-p0: preserve-alm-gen-p0
stash@{3}: On mattergen-lora-p0: preserve-mattergen-lora-p0
stash@{4}: On composable-property-adapter-p0: preserve-gcpa-p0

```

## git worktree list --porcelain

```text
worktree /mnt/datasets-livsyn/dxl/mattergen_v1
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/raab-sc-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_adaptive_neighbor
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/adaptive-neighborhood-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/field-async-freeze-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_cell_atom_fusion
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/cell-atom-fusion-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_closed_loop_guidance
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/closed-loop-force-guidance-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_field_schedule
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/field-time-schedule-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_fppc
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/fp-pc-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/mattersim-late-force-guidance-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_self_condition
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/multifield-self-conditioning-p0

worktree /mnt/datasets-livsyn/dxl/mattergen_v1_self_correcting_search
HEAD 1af301c5e534ff6b9b9f96ea5dab48036d9c46af
branch refs/heads/experiment/self-correcting-search-p0


```

## git branch -a

```text
  analysis/tcl-formal256-root-cause
+ experiment/adaptive-neighborhood-p0
  experiment/alm-gen-p0
+ experiment/cell-atom-fusion-p0
+ experiment/closed-loop-force-guidance-p0
  experiment/composable-property-adapter-p0
  experiment/corrector-residual-distillation-formal256
  experiment/corrector-residual-distillation-v1
  experiment/corrector-residual-distillation-v2
  experiment/corrector-residual-distillation-v3-anchor
  experiment/cross-field-interaction-p0
  experiment/cross-field-interaction-p1
  experiment/distribution-balanced-quality-adapter
  experiment/distribution-constrained-quality-adapter-p0
  experiment/distribution-constrained-quality-adapter-p1
  experiment/distribution-constrained-quality-adapter-p1b
+ experiment/field-async-freeze-p0
+ experiment/field-time-schedule-p0
+ experiment/fp-pc-p0
  experiment/gbsa-tcl-p0
  experiment/gbsa-tcl-p1
  experiment/gbsa-tcl-prospective32
  experiment/global-transformer-adapter-p0
  experiment/mattergen-lora-p0
  experiment/mattergen-rl-p0
+ experiment/mattersim-late-force-guidance-p0
+ experiment/multifield-self-conditioning-p0
* experiment/raab-sc-p0
+ experiment/self-correcting-search-p0
  experiment/tcl-dml-p0
  experiment/tcl-formal256
  experiment/tcl-p1
  experiment/tcl-p2
  main
  remotes/origin/HEAD -> origin/main
  remotes/origin/archive/thesis-analysis-package-v1
  remotes/origin/experiment/a0-e3g-leakage-diagnostic256
  remotes/origin/feature/a0-e3g-compatibility64
  remotes/origin/feature/a0-e3g-formal256
  remotes/origin/feature/a0-e3g-independent64
  remotes/origin/feature/budget-aware-corrector-gating
  remotes/origin/feature/cg-tdr
  remotes/origin/feature/chapter-evidence-packs-v1
  remotes/origin/feature/convergence-aware-corrector-gating
  remotes/origin/feature/crystalrepa-repro
  remotes/origin/feature/fn-pra
  remotes/origin/feature/gemnet-fused-inference-fastgate
  remotes/origin/feature/mps-runtime-fastgate
  remotes/origin/feature/postgen-quality-modules-fastgate
  remotes/origin/feature/q3-e3-pcr-formal256
  remotes/origin/feature/q3-e3-pcr-frozen64
  remotes/origin/feature/rp-qtfg
  remotes/origin/feature/spg-mattergen-fastgate
  remotes/origin/feature/spg-static-periodic-graph-mvp
  remotes/origin/feature/thesis-planning-figures-v1
  remotes/origin/feature/thesis-positioning-chapter3-v1
  remotes/origin/main
  remotes/upstream/main

```

## nvidia-smi

```text
Mon Sep 14 20:17:48 2026       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H20                     On  | 00000000:65:02.0 Off |                    0 |
| N/A   47C    P0             312W / 500W |  31086MiB / 97871MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H20                     On  | 00000000:65:03.0 Off |                    0 |
| N/A   56C    P0             334W / 500W |  47909MiB / 97871MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA H20                     On  | 00000000:67:02.0 Off |                    0 |
| N/A   41C    P0             118W / 500W |  83319MiB / 97871MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA H20                     On  | 00000000:67:03.0 Off |                    0 |
| N/A   49C    P0             325W / 500W |  31221MiB / 97871MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   4  NVIDIA H20                     On  | 00000000:69:02.0 Off |                    0 |
| N/A   48C    P0             318W / 500W |  31017MiB / 97871MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   5  NVIDIA H20                     On  | 00000000:69:03.0 Off |                    0 |
| N/A   37C    P0              75W / 500W |      3MiB / 97871MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   6  NVIDIA H20                     On  | 00000000:6B:02.0 Off |                    0 |
| N/A   29C    P0              73W / 500W |      3MiB / 97871MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   7  NVIDIA H20                     On  | 00000000:6B:03.0 Off |                    0 |
| N/A   27C    P0              72W / 500W |      3MiB / 97871MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    456212      G   ...a0/acone-mujoco-sim/venv/bin/python       66MiB |
|    0   N/A  N/A   1923581      C   ...VIRONMENTS/plenoptic-h20/bin/python    31010MiB |
|    1   N/A  N/A   1923579      C   ...VIRONMENTS/plenoptic-h20/bin/python    47902MiB |
|    2   N/A  N/A   3729946      C   .../WM-RM/pi05/openpi/.venv/bin/python    83308MiB |
|    3   N/A  N/A   1725836      G   ...libero-agent-slice/.venv/bin/python      201MiB |
|    3   N/A  N/A   1923582      C   ...VIRONMENTS/plenoptic-h20/bin/python    31010MiB |
|    4   N/A  N/A   1923580      C   ...VIRONMENTS/plenoptic-h20/bin/python    31010MiB |
+---------------------------------------------------------------------------------------+

```

## tmux ls

```text
no server running on /tmp/tmux-0/default

```

## df -h /mnt/datasets-livsyn /tmp

```text
Filesystem               Size  Used Avail Use% Mounted on
JuiceFS:datasets-livsyn  1.0P  5.5T 1019T   1% /mnt/datasets-livsyn
/dev/vda2                 49G   29G   19G  61% /

```

## free -h

```text
               total        used        free      shared  buff/cache   available
Mem:           1.9Ti        76Gi        55Gi       138Gi       1.8Ti       1.7Ti
Swap:             0B          0B          0B

```

## ps -eo pid,ppid,user,lstart,args

```text
   4255       1 root     Wed Sep  9 11:19:20 2026 python3 /usr/bin/irqbalance-ng -c /etc/default/irqbalance-ng
   4256       1 root     Wed Sep  9 11:19:20 2026 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers
   4330       1 root     Wed Sep  9 11:19:20 2026 /usr/bin/python3 -Es /usr/sbin/tuned -l -P
 177845       1 root     Mon Sep 14 13:52:23 2026 sh /root/.vscode-server/cli/servers/Stable-645f29cc3176500b4b5762ba887cf2a7f0ffdf2c/server/bin/code-server --connection-token=remotessh --accept-server-license-terms --install-extension=ms-python.python --install-extension=ms-toolsai.jupyter-renderers --install-extension=ms-toolsai.jupyter-keymap --install-extension=ms-toolsai.jupyter --agent-host-bridge-port=33951 --agent-host-bridge-host=127.0.0.1 --start-server --enable-remote-auto-shutdown --socket-path=/tmp/code-b87cf0a6-3bf5-461b-9fe7-28884e964370
 177850  177845 root     Mon Sep 14 13:52:23 2026 /root/.vscode-server/cli/servers/Stable-645f29cc3176500b4b5762ba887cf2a7f0ffdf2c/server/node /root/.vscode-server/cli/servers/Stable-645f29cc3176500b4b5762ba887cf2a7f0ffdf2c/server/out/server-main.js --connection-token=remotessh --accept-server-license-terms --install-extension=ms-python.python --install-extension=ms-toolsai.jupyter-renderers --install-extension=ms-toolsai.jupyter-keymap --install-extension=ms-toolsai.jupyter --agent-host-bridge-port=33951 --agent-host-bridge-host=127.0.0.1 --start-server --enable-remote-auto-shutdown --socket-path=/tmp/code-b87cf0a6-3bf5-461b-9fe7-28884e964370
 456212       1 root     Mon Sep 14 14:40:43 2026 /data0/acone-mujoco-sim/venv/bin/python /data0/acone-mujoco-sim/venv/bin/acone-mujoco-service --host 127.0.0.1 --driver-port 18030 --camera-port 18031 --admin-port 18032 --admin-token-file /data0/acone-mujoco-sim/run/admin-token
 723775       1 root     Mon Sep 14 15:36:53 2026 /data0/acone-mujoco-viewer/venv/bin/python3 /data0/acone-mujoco-viewer/venv/bin/acone-mujoco-viewer --model /data0/acone-mujoco-sim/current/src/acone_mujoco_sim/models/acone_dual.xml --admin-url http://127.0.0.1:18032 --admin-token-file /data0/acone-mujoco-sim/run/admin-token --host 127.0.0.1 --port 8080 --poll-hz 20 --status-file /data0/acone-mujoco-viewer/run/status.json
 729051  728178 root     Mon Sep 14 15:37:51 2026 /root/.vscode-server/extensions/ms-python.vscode-python-envs-1.36.0-linux-x64/python-env-tools/bin/pet server
 729877  728178 root     Mon Sep 14 15:37:56 2026 /root/.vscode-server/cli/servers/Stable-645f29cc3176500b4b5762ba887cf2a7f0ffdf2c/server/node /root/.vscode-server/extensions/ms-python.vscode-pylance-2026.3.1/dist/server.bundle.js --cancellationReceive=file:5e7f32bb4a8d176452ce956d92f73d1d250d72faf2 --node-ipc --clientProcessId=728178
1075949       1 root     Sat Sep 12 20:58:05 2026 /usr/bin/python3 /mnt/datasets-livsyn/llq/new_datasets/tools/downloadrelay.py
1578028       1 root     Mon Sep 14 18:49:49 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/plenoptic-repro/tools/prepare_tabletop.py --config /mnt/datasets-livsyn/llq/plenoptic-repro/configs/plenoptic/tabletop_prepare_taco_all12.json
1578956       1 root     Mon Sep 14 18:50:01 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/plenoptic-repro/tools/prepare_hoi_supervision.py --config /mnt/datasets-livsyn/llq/plenoptic-repro/configs/plenoptic/hoi_supervision_prepare_taco_all12.json --datasets taco
1658812       1 root     Mon Sep 14 19:07:46 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/plenoptic-repro/tools/hoi_validation_service.py serve
1659178 1658812 root     Mon Sep 14 19:07:49 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659480 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659482 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659484 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659486 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659488 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659490 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659492 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659494 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659497 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659499 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659501 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659503 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659506 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659508 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659510 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659512 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659514 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659516 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659518 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659520 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659522 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659524 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659526 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659531 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659533 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659535 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659537 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659539 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659541 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659543 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659546 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1659548 1659178 root     Mon Sep 14 19:07:53 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python3 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1658812 --read-fd=3 --write-fd=6
1725836       1 root     Mon Sep 14 19:22:48 2026 /data0/libero-agent-slice/.venv/bin/python -m libero_env.service --policy-host 127.0.0.1 --policy-port 8000 --port 18040 --evaluator-token-file /data0/libero-agent-slice/secrets/assembly-evaluator-token
1729094       1 root     Mon Sep 14 19:23:25 2026 /data0/acone-mujoco-viewer/venv/bin/python /data0/libero-agent-slice/dashboard/live_viewer.py --state-url http://127.0.0.1:18041 --token-file /data0/libero-agent-slice/secrets/assembly-evaluator-token --host 127.0.0.1 --port 8082 --poll-hz 30
1918257 1893845 root     Mon Sep 14 20:06:35 2026 python tools/validation_runner.py start --synchronized-latest --checkpoint ../OUTPUTS/basic_stage1_24gpu/step-016000.pt --standard-only --gpus 0,1,3,4 --wait
1918263 1918257 root     Mon Sep 14 20:06:36 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/plenoptic-repro/tools/validation_runner.py _run
1923385 1918263 root     Mon Sep 14 20:07:46 2026 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29691 tools/evaluate_stage1.py --checkpoint /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/checkpoint.pt --output /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1 --video-batch-size 1 --publish-progress --custom-suite /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/empty-custom-suite.json
1923579 1923385 root     Mon Sep 14 20:07:48 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python -u tools/evaluate_stage1.py --checkpoint /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/checkpoint.pt --output /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1 --video-batch-size 1 --publish-progress --custom-suite /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/empty-custom-suite.json
1923580 1923385 root     Mon Sep 14 20:07:48 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python -u tools/evaluate_stage1.py --checkpoint /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/checkpoint.pt --output /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1 --video-batch-size 1 --publish-progress --custom-suite /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/empty-custom-suite.json
1923581 1923385 root     Mon Sep 14 20:07:48 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python -u tools/evaluate_stage1.py --checkpoint /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/checkpoint.pt --output /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1 --video-batch-size 1 --publish-progress --custom-suite /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/empty-custom-suite.json
1923582 1923385 root     Mon Sep 14 20:07:48 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python -u tools/evaluate_stage1.py --checkpoint /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/checkpoint.pt --output /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1 --video-batch-size 1 --publish-progress --custom-suite /mnt/datasets-livsyn/llq/OUTPUTS/validation/.work/20260914-200636-331307-h20-1/empty-custom-suite.json
1923825 1923581 root     Mon Sep 14 20:07:52 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1923826 1923580 root     Mon Sep 14 20:07:52 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1923827 1923582 root     Mon Sep 14 20:07:52 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1923828 1923579 root     Mon Sep 14 20:07:52 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924102 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924104 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924106 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924108 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924111 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924113 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924114 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924117 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924118 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924120 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924122 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924124 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924126 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924130 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924131 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924133 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924135 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924137 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924140 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924142 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924144 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924146 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924148 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924150 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924152 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924154 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924156 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924158 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924160 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924164 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924165 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924168 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924169 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924172 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924173 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924176 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924177 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924180 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924181 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924184 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924185 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924188 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924189 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924192 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924193 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924196 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924197 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924201 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924202 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924205 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924206 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924209 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924210 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924213 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924214 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924217 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924218 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924221 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924223 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924225 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924227 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924229 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924234 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924236 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924237 1923827 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923582 --read-fd=3 --write-fd=6
1924240 1923826 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923580 --read-fd=3 --write-fd=6
1924241 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924249 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924251 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924252 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924254 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924257 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924258 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924262 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924263 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924266 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924267 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924270 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924271 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924273 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924275 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924277 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924279 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924281 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924283 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924286 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924288 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924291 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924292 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924295 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924296 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924298 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924300 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924302 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924304 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924306 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924308 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924310 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924312 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924314 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924316 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924318 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924320 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924323 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924325 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924327 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924329 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924331 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924334 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924335 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924338 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924339 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924342 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924343 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924346 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924347 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924350 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924351 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924354 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924355 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924358 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924359 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924362 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924364 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924366 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924369 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1924372 1923825 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923581 --read-fd=3 --write-fd=6
1924377 1923828 root     Mon Sep 14 20:07:56 2026 /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/bin/python /mnt/datasets-livsyn/llq/ENVIRONMENTS/plenoptic-h20/lib/python3.10/site-packages/torch/_inductor/compile_worker/__main__.py --pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler --kind=fork --workers=32 --parent=1923579 --read-fd=3 --write-fd=6
1968571 1934457 root     Mon Sep 14 20:17:48 2026 /usr/bin/python3 experiments/final_thesis_results/audit_environment.py
3729946       1 root     Sat Sep 12 12:09:06 2026 /data0/WM-RM/pi05/openpi/.venv/bin/python /data0/libero-agent-slice/deploy/serve_policy_loopback.py --env LIBERO --port 8000
```
