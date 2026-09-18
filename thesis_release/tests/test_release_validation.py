"""CPU-only regression tests for the thesis release."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_release.py"
SPEC = importlib.util.spec_from_file_location("validate_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


class ReleaseMechanismTests(unittest.TestCase):
    def test_all_guidance_two_reference_equivalence(self) -> None:
        VALIDATE.test_reference_equivalence()

    def test_coordinate_transform_and_force_update(self) -> None:
        VALIDATE.test_coordinate_and_force_helpers()

    def test_reconstructed_allocator(self) -> None:
        VALIDATE.check_reconstruction()

    def test_seed_manifests_and_configs(self) -> None:
        VALIDATE.check_seed_manifests_and_configs()


if __name__ == "__main__":
    unittest.main()
