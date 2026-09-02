"""Inspect an OPV2V split before training or inference."""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitSummary:
    scenes: int
    agents: int
    lidar_frames: int
    label_frames: int
    paired_frames: int


def inspect_split(split_root: Path) -> SplitSummary:
    if not split_root.is_dir():
        raise FileNotFoundError(f"OPV2V split directory not found: {split_root}")

    scene_count = agent_count = lidar_count = label_count = paired_count = 0
    for scene in sorted(path for path in split_root.iterdir() if path.is_dir()):
        scene_count += 1
        for agent in sorted(path for path in scene.iterdir() if path.is_dir()):
            agent_count += 1
            lidar_stems = {path.stem for path in agent.glob("*.pcd")}
            label_stems = {path.stem for path in agent.glob("*.yaml")}
            lidar_count += len(lidar_stems)
            label_count += len(label_stems)
            paired_count += len(lidar_stems & label_stems)

    return SplitSummary(scene_count, agent_count, lidar_count, label_count, paired_count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OPV2V scene/agent/frame structure")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()

    found = False
    totals = Counter()
    for split_name in ("train", "validate", "test"):
        split_root = args.dataset_root / split_name
        if not split_root.is_dir():
            continue
        found = True
        summary = inspect_split(split_root)
        print(
            f"{split_name}: scenes={summary.scenes}, agents={summary.agents}, "
            f"pcd={summary.lidar_frames}, yaml={summary.label_frames}, "
            f"paired={summary.paired_frames}"
        )
        totals.update(summary.__dict__)

    if not found:
        print("No train/validate/test directory found under dataset root.")
        return 1
    if totals["paired_frames"] == 0:
        print("No paired .pcd/.yaml frames found.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
