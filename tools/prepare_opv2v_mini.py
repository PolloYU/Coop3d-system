"""Create a tiny, ego-only OPV2V split and normalize legacy matrix poses."""

from __future__ import print_function

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import yaml


def matrix_to_pose(matrix):
    """Inverse of OpenCOOD transformation_utils.x_to_world."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        return matrix.tolist()
    pitch = math.asin(max(-1.0, min(1.0, matrix[2, 0])))
    roll = math.atan2(-matrix[2, 1], matrix[2, 2])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]),
            math.degrees(roll), math.degrees(yaw), math.degrees(pitch)]


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {plain(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted OPV2V test split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    scenarios = sorted(path for path in args.source.iterdir() if path.is_dir())
    if not scenarios:
        raise ValueError("No scenario directory found: {}".format(args.source))
    scenario = scenarios[0]
    cavs = sorted((path for path in scenario.iterdir() if path.is_dir()),
                  key=lambda path: (int(path.name) < 0, int(path.name)))
    if not cavs:
        raise ValueError("No CAV directory found: {}".format(scenario))
    ego = cavs[0]
    yaml_files = sorted(path for path in ego.glob("*.yaml")
                        if "additional" not in path.name)[:args.frames]
    target = args.output / scenario.name / ego.name
    target.mkdir(parents=True, exist_ok=True)

    copied = 0
    for yaml_path in yaml_files:
        pcd_path = yaml_path.with_suffix(".pcd")
        if not pcd_path.is_file():
            continue
        with yaml_path.open("r") as stream:
            record = yaml.load(stream, Loader=yaml.Loader)
        for key in ("lidar_pose", "true_ego_pos"):
            if key in record:
                record[key] = matrix_to_pose(record[key])
        record = plain(record)
        with (target / yaml_path.name).open("w") as stream:
            yaml.safe_dump(record, stream, default_flow_style=False,
                           sort_keys=False)
        shutil.copy2(str(pcd_path), str(target / pcd_path.name))
        copied += 1

    if not copied:
        raise ValueError("No paired YAML/PCD frames found")
    print("Created {} ego-only frames: {}".format(copied, args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
