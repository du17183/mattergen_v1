"""Run the exact P0 CHGNet property evaluator on the Formal32 cohort."""
from pathlib import Path

from experiments.mattersim_late_force_guidance_p0 import evaluate_p0_properties as p0


ROOT = Path(__file__).resolve().parent
p0.ROOT = ROOT
p0.SEEDS = tuple(range(730000, 730032))
p0.METHODS = ("C0", "F0")


if __name__ == "__main__":
    internal = ROOT / "p0_structures"
    if internal.exists():
        if not internal.is_symlink() or internal.resolve() != (ROOT / "structures").resolve():
            raise FileExistsError(internal)
    else:
        internal.symlink_to(ROOT / "structures", target_is_directory=True)
    p0.main()
    (ROOT / "p0_property_metrics.csv").replace(ROOT / "property_metrics.csv")
    (ROOT / "p0_property_summary.json").replace(ROOT / "property_summary.json")
