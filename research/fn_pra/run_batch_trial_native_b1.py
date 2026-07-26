"""Run the batch_size=1 benchmark through the unmodified native A0 loop."""

from __future__ import annotations

from research.fn_pra import independent_batch
from research.fn_pra.native_single_batch import NativeSingleTrajectoryGuidedPredictorCorrector
from research.fn_pra.run_batch_trial import main


independent_batch.IndependentTrajectoryGuidedPredictorCorrector = (
    NativeSingleTrajectoryGuidedPredictorCorrector
)


if __name__ == "__main__":
    raise SystemExit(main())
