import tempfile
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.single_detection.contracts import Detection3D, DetectionResult
from models.single_detection.opencood_runner import OpenCoodConfig, OpenCoodRunner
from tools.inspect_opv2v import inspect_split
from tools.infer_single_opv2v import force_ego_only


class DetectionContractTests(unittest.TestCase):
    def test_result_can_be_serialized(self):
        detection = Detection3D("vehicle", 0.9, (1, 2, 0, 4, 2, 1.5, 0.1))
        result = DetectionResult("scene-1", "0001", "ego", [detection])
        self.assertEqual(result.to_dict()["detections"][0]["label"], "vehicle")

    def test_invalid_score_is_rejected(self):
        with self.assertRaises(ValueError):
            Detection3D("vehicle", 1.1, (1, 2, 0, 4, 2, 1.5, 0.1))


class OpenCoodRunnerTests(unittest.TestCase):
    def test_builds_official_inference_command(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / "opencood" / "tools" / "inference.py"
            script.parent.mkdir(parents=True)
            script.touch()
            model_dir = root / "checkpoint"
            model_dir.mkdir()
            (model_dir / "config.yaml").touch()
            (model_dir / "net_epoch_bestval_at1.pth").touch()
            python = root / "python.exe"
            python.touch()

            runner = OpenCoodRunner(OpenCoodConfig(root, model_dir, python))
            self.assertEqual(runner.problems(), [])
            command = runner.build_command()
            self.assertIn("--model_dir", command)
            self.assertIn("--save_npy", command)


class Opv2vInspectionTests(unittest.TestCase):
    def test_counts_only_paired_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            agent = Path(temp) / "scene-1" / "ego"
            agent.mkdir(parents=True)
            (agent / "0001.pcd").touch()
            (agent / "0001.yaml").touch()
            (agent / "0002.pcd").touch()

            summary = inspect_split(Path(temp))
            self.assertEqual(summary.scenes, 1)
            self.assertEqual(summary.agents, 1)
            self.assertEqual(summary.lidar_frames, 2)
            self.assertEqual(summary.paired_frames, 1)

    def test_single_config_forces_one_ego_agent(self):
        config = {
            "validate_dir": "old",
            "train_params": {"max_cav": 7},
            "fusion": {"core_method": "LateFusionDataset"},
        }
        result = force_ego_only(config, Path("test"))
        self.assertEqual(result["train_params"]["max_cav"], 1)
        self.assertEqual(result["validate_dir"], "test")
        self.assertFalse(result["wild_setting"]["async"])


if __name__ == "__main__":
    unittest.main()
