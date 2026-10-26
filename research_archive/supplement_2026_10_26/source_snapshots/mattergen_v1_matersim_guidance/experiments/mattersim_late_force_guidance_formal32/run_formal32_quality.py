"""Run the exact P0 official Alex-MP quality protocol for Formal32."""
from pathlib import Path

from experiments.mattersim_late_force_guidance_p0 import run_p0_quality as p0


ROOT = Path(__file__).resolve().parent
p0.ROOT = ROOT
p0.METHODS = ("C0", "F0")


if __name__ == "__main__":
    internal = ROOT / "p0_structures"
    if not internal.exists():
        internal.symlink_to(ROOT / "structures", target_is_directory=True)
    p0.main()
