"""Apply the P0 aggregation/property implementation to authorized Formal256."""

from pathlib import Path
import prepare_fresh_p0 as implementation

implementation.P0 = Path(__file__).resolve().parent / "formal256"

if __name__ == "__main__":
    implementation.main()
