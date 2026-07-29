"""Repository-relative paths used by every thesis generator."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
THESIS_ROOT = REPO_ROOT / "thesis"
ARCHIVE_ROOT = REPO_ROOT / "thesis_archive"
FIGURE_ROOT = THESIS_ROOT / "figures"
FIGURE_OUTPUTS = FIGURE_ROOT / "generated"
FIGURE_SOURCE_DATA = FIGURE_ROOT / "source_data"
TABLE_ROOT = THESIS_ROOT / "tables"


def ensure_output_directories() -> None:
    """Create only generated-artifact directories, never source-data archives."""
    for path in (
        FIGURE_OUTPUTS / "pdf",
        FIGURE_OUTPUTS / "svg",
        FIGURE_OUTPUTS / "png",
        FIGURE_SOURCE_DATA,
        FIGURE_ROOT / "validation",
        TABLE_ROOT / "csv",
        TABLE_ROOT / "xlsx",
        TABLE_ROOT / "markdown",
        TABLE_ROOT / "latex",
        TABLE_ROOT / "validation",
    ):
        path.mkdir(parents=True, exist_ok=True)

