"""Technical GPU reassignment wrapper for the frozen Oracle quality evaluator.

The scientific evaluator and six group definitions remain imported unchanged
from ``run_oracle_quality.py``.  Only physical GPU placement is replaced after
GPU 0 became occupied by an unrelated user task.
"""

from __future__ import annotations

import run_oracle_quality as frozen_quality


REASSIGNED_GPU_POOL = (1, 3, 4, 5, 6, 7)


if __name__ == "__main__":
    frozen_quality.GPU_POOL = REASSIGNED_GPU_POOL
    frozen_quality.main()
