"""Check archive boundaries before publishing generated tensor data."""
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).with_name("build_snapshot.py")
SPEC = importlib.util.spec_from_file_location("snapshot_builder", PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class SnapshotBoundaryTests(unittest.TestCase):
    def test_generated_tensor_states_are_retained(self):
        for name in [
            "experiments/confirmation/generation/1/prefix_checkpoint.pt",
            "experiments/diagnostic/diagnostic_states/1/t_0.02.pt",
            "experiments/distillation/teacher_data/train/1/shard_00001.pt",
            "experiments/distillation/dagger_data/train/1/shard_00001.pt",
            "experiments/alm/structures/stage3a_inference_meta.pt",
        ]:
            with self.subTest(path=name):
                self.assertIsNone(BUILDER.file_policy(name))

    def test_network_weights_and_caches_are_excluded(self):
        for name in [
            "experiments/method/checkpoints/model.pt",
            "experiments/method/adapter.pt",
            "experiments/method/checkpoint.ckpt",
            "experiments/method/weights.safetensors",
            "experiments/method/chgnet.pth.tar",
            "experiments/method/runtime_tmp/random.py",
            "experiments/method/__pycache__/model.pyc",
            "experiments/method/tmp/reference_MP2020correction",
        ]:
            with self.subTest(path=name):
                self.assertIsNotNone(BUILDER.file_policy(name))

    def test_secret_scan_returns_only_path_and_label(self):
        fake = ("ghp_" + "A" * 36).encode()
        findings = BUILDER.scan_bytes(fake, "example.log")
        self.assertEqual(findings, [{"path": "example.log", "pattern": "github_token"}])
        self.assertNotIn(fake.decode(), str(findings))
        self.assertEqual(BUILDER.scan_bytes(b"checkpoint path: hf_mattergen", "run.log"), [])


if __name__ == "__main__":
    unittest.main()
