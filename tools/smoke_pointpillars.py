"""Build OpenCOOD PointPillars without requiring data or a checkpoint."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencood-root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    root = args.opencood_root.resolve()
    config = args.config or root / "opencood" / "hypes_yaml" / "point_pillar_late_fusion.yaml"
    sys.path.insert(0, str(root))

    import torch
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    hypes = load_yaml(str(config))
    model = create_model(hypes).to(args.device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("PointPillars model: OK")
    print("Config: {}".format(config))
    print("Device: {}".format(args.device))
    print("Parameters: {:,}".format(parameter_count))
    if args.device == "cuda":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
