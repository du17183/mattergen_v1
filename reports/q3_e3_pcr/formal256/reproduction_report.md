# Reproduce E3-PCR Formal 256

```bash
source /data/dxl/env.sh
cd /data/dxl/mattergen_v1
git switch feature/q3-e3-pcr-formal256
/data/dxl/tools/q3_e3_pcr/formal256/resume.sh
/data/dxl/tools/q3_e3_pcr/formal256/status.sh
```

The runner is resumable. Successful generation and relaxation tasks are hash-validated and are not rerun.
