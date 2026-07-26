"""V1-D entry point with decision-specific progress and report routing."""

from __future__ import annotations

from pathlib import Path

from research.fn_pra import train_v1 as base
from research.fn_pra import train_v1_decision as decision


_set_stage = base.set_stage
_PathType = type(Path())


class DecisionReportPath(_PathType):
    def __truediv__(self, key):
        if key == "v1_smoke_training_summary.json":
            key = "v1_decision_training_summary.json"
        return super().__truediv__(key)


def decision_set_stage(stage: str, status: str, detail: str, metrics=None) -> None:
    if stage == "v1_smoke_training":
        stage = "v1_decision_training"
        detail = detail.replace("V1-S", "V1-D").replace(
            "Static REPA V1 smoke training", "Static REPA V1 decision training"
        )
    _set_stage(stage, status, detail, metrics)


def main() -> None:
    base.set_stage = decision_set_stage
    base.REPORTS = DecisionReportPath(str(base.REPORTS))
    decision.main()


if __name__ == "__main__":
    main()
