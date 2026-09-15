"""Apply the frozen three-policy quality workflow to authorized Formal256."""

from pathlib import Path
import run_p0_quality as implementation

implementation.P0 = Path(__file__).resolve().parent / "formal256"

if __name__ == "__main__":
    implementation.main()
