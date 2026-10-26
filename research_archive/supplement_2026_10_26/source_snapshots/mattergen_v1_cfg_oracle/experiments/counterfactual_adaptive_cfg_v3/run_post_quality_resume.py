"""Resume the frozen gate chain after all six quality groups complete."""

from __future__ import annotations

import json

from run_frozen_pipeline import (
    ORACLE,
    ROOT,
    PYTHON,
    run,
    status,
    write_final_report,
)


def main() -> None:
    try:
        for group in ("SHORT_DOWN", "SHORT_KEEP", "SHORT_UP", "FULL_DOWN", "FULL_KEEP", "FULL_UP"):
            if not (ORACLE / "quality" / group / "official_detailed.json.gz").is_file():
                raise RuntimeError(f"quality group incomplete: {group}")
        if (ORACLE / "oracle_summary.json").exists():
            raise FileExistsError("refusing to overwrite an existing Oracle gate result")
        status("oracle_headroom_analysis_resumed")
        run(
            [PYTHON, str(ROOT / "run_analysis_with_column_fix.py")],
            environment={
                "MPLCONFIGDIR": str(ROOT / ".matplotlib"),
                "PYTHONUNBUFFERED": "1",
            },
            log_name="oracle_analysis_resume.log",
        )
        headroom = json.loads((ORACLE / "oracle_summary.json").read_text())
        if headroom["ORACLE_HEADROOM"] != "GO":
            write_final_report(headroom, None)
            status("stopped_after_oracle_headroom", "COMPLETED", decision=headroom)
            return
        status("oracle_predictability")
        run(
            [PYTHON, str(ROOT / "run_predictability.py")],
            environment={
                "MPLCONFIGDIR": str(ROOT / ".matplotlib"),
                "PYTHONUNBUFFERED": "1",
            },
            log_name="predictability.log",
        )
        predictability = json.loads((ROOT / "controller/decision_summary.json").read_text())
        if predictability["ORACLE_PREDICTABILITY"] != "GO":
            write_final_report(headroom, predictability)
            status("stopped_after_predictability", "COMPLETED", decision=predictability)
            return
        status(
            "v3_implementation_required",
            "GATE_GO",
            decision=predictability,
            note="Both frozen scientific gates passed; implement sparse V3 before fresh P0.",
        )
    except BaseException as error:
        status("technical_failure", "FAILED", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
