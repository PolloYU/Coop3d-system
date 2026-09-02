"""Check whether this machine is ready for PointPillars/OpenCOOD."""

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import List


def module_version(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "missing"
    module = __import__(name)
    return str(getattr(module, "__version__", "installed"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencood-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    blockers = []  # type: List[str]
    warnings = []  # type: List[str]

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 7):
        warnings.append("This OpenCOOD revision declares Python 3.7.11; use a separate environment.")

    torch_version = module_version("torch")
    spconv_version = module_version("spconv")
    print(f"PyTorch: {torch_version}")
    print(f"spconv: {spconv_version}")
    if torch_version == "missing":
        blockers.append("PyTorch is not installed in this Python environment.")
    if spconv_version == "missing":
        blockers.append("spconv is not installed in this Python environment.")

    if importlib.util.find_spec("torch") is not None:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.opencood_root is not None:
        script = args.opencood_root / "opencood" / "tools" / "inference.py"
        print(f"OpenCOOD inference.py: {'ok' if script.is_file() else 'missing'} ({script})")
        if not script.is_file():
            blockers.append("OpenCOOD inference.py was not found.")

    if args.model_dir is not None:
        config = args.model_dir / "config.yaml"
        checkpoints = list(args.model_dir.glob("*.pth")) if args.model_dir.is_dir() else []
        print(f"Model config: {'ok' if config.is_file() else 'missing'} ({config})")
        print(f"Model checkpoints: {len(checkpoints)} .pth file(s)")
        if not config.is_file() or not checkpoints:
            blockers.append("Model directory needs config.yaml and at least one .pth checkpoint.")

    if args.data_root is not None:
        splits = [name for name in ("train", "validate", "test") if (args.data_root / name).is_dir()]
        print(f"Dataset splits: {', '.join(splits) if splits else 'missing'} ({args.data_root})")
        if not splits:
            blockers.append("No train/validate/test split was found under the dataset root.")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for blocker in blockers:
        print(f"BLOCKER: {blocker}")
    print(f"OpenCOOD runtime: {'READY' if not blockers else 'NOT READY'}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
