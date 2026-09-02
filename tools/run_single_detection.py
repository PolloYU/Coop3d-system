"""Validate or launch the current OpenCOOD baseline."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.single_detection.opencood_runner import OpenCoodConfig, OpenCoodRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencood-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute", action="store_true", help="Actually run inference")
    args = parser.parse_args()

    runner = OpenCoodRunner(OpenCoodConfig(
        opencood_root=args.opencood_root.resolve(),
        model_dir=args.model_dir.resolve(),
        python_executable=args.python_executable.resolve(),
    ))
    problems = runner.problems()
    if problems:
        print("Environment is not ready:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Command:")
    print(" ".join(f'\"{part}\"' if " " in part else part for part in runner.build_command()))
    if args.execute:
        runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
