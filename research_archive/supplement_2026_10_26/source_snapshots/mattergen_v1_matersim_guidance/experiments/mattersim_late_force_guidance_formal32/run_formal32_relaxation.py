"""Run the exact P0 MatterSim relaxation protocol for Formal32."""
from pathlib import Path

from experiments.mattersim_late_force_guidance_p0 import run_p0_relaxation as p0


ROOT = Path(__file__).resolve().parent
p0.ROOT = ROOT
p0.JOBS = (("C0", 5), ("F0", 6))


if __name__ == "__main__":
    internal = ROOT / "p0_structures"
    if not internal.exists():
        internal.symlink_to(ROOT / "structures", target_is_directory=True)
    p0.main()
