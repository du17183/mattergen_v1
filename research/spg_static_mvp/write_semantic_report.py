from __future__ import annotations

from research.spg_static_mvp.common import PROJECT, REPORTS, atomic_text, set_stage


def main() -> int:
    source = PROJECT / "research/spg_static_mvp/original_graph_semantics.md"
    destination = REPORTS / "original_graph_semantics.md"
    atomic_text(destination, source.read_text(encoding="utf-8"))
    set_stage(
        "semantic_code_map",
        "success",
        "Mapped exact PBC, top-50, symmetric-edge, and triplet semantics.",
        {
            "report": str(destination),
            "reference_commit": "9bc6747a3ddfd26db6d931bcdb6df5d299844544",
            "torch": "2.7.1+cu128",
            "equal_distance_policy": "eager fallback when a tie crosses top-50",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
