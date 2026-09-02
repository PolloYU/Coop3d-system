"""Run strict ego-only PointPillars inference on a small OPV2V split."""

import argparse
import json
from pathlib import Path
import sys
import time


def force_ego_only(hypes, data_root):
    """Return an OpenCOOD config constrained to the first (ego) CAV."""
    hypes["validate_dir"] = str(data_root)
    hypes.setdefault("train_params", {})["max_cav"] = 1
    hypes["fusion"]["core_method"] = "LateFusionDataset"
    hypes.setdefault("wild_setting", {})["async"] = False
    hypes["wild_setting"]["loc_err"] = False
    return hypes


def verify_ego_only(dataset):
    if dataset.max_cav != 1:
        raise RuntimeError("Single-agent mode requires max_cav=1")
    for scenario in dataset.scenario_database.values():
        agents = list(scenario.values())
        if len(agents) != 1 or not agents[0].get("ego", False):
            raise RuntimeError("Dataset contains a non-ego input in single-agent mode")


def lidar_to_numpy(origin_lidar, np):
    if isinstance(origin_lidar, (list, tuple)):
        origin_lidar = origin_lidar[0]
    if hasattr(origin_lidar, "detach"):
        origin_lidar = origin_lidar.detach().cpu().numpy()
    origin_lidar = np.asarray(origin_lidar)
    if origin_lidar.ndim == 3 and origin_lidar.shape[0] == 1:
        origin_lidar = origin_lidar[0]
    if origin_lidar.ndim != 2 or origin_lidar.shape[1] < 3:
        raise ValueError("Unexpected origin_lidar shape: {}".format(
            origin_lidar.shape))
    return origin_lidar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencood-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="One OPV2V split, for example .../test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/single_3d"))
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="Show each frame immediately after real inference")
    parser.add_argument("--live-delay", type=float, default=0.35,
                        help="Minimum seconds to display each inferred frame")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    if args.max_frames < 1:
        raise ValueError("max-frames must be at least 1")
    root = args.opencood_root.resolve()
    model_dir = args.model_dir.resolve()
    data_root = args.data_root.resolve()
    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Missing model config: {}".format(config_path))
    if not data_root.is_dir():
        raise FileNotFoundError("Missing OPV2V split: {}".format(data_root))

    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import box_utils

    hypes = force_ego_only(load_yaml(str(config_path)), data_root)
    dataset = build_dataset(hypes, visualize=True, train=False)
    verify_ego_only(dataset)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model.eval()

    live = None
    live_latencies = []
    if args.live:
        import matplotlib.pyplot as plt
        from matplotlib import transforms
        from matplotlib.patches import Rectangle
        plt.ion()
        figure, axis = plt.subplots(figsize=(13, 7))
        figure.canvas.manager.set_window_title("Coop3D Live Inference")
        if args.fullscreen:
            figure.canvas.manager.full_screen_toggle()
        axis.axis("off")
        axis.text(0.5, 0.5, "PointPillars loaded\nWaiting for the first frame...",
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=20, weight="bold")
        figure.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.05)
        live = (plt, transforms, Rectangle, figure, axis)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for index, batch_data in enumerate(loader):
        if index >= args.max_frames:
            break
        batch_data = train_utils.to_device(batch_data, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.time()
        with torch.no_grad():
            pred_corners, pred_scores, gt_corners = inference_utils.inference_late_fusion(
                batch_data, model, dataset)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - started) * 1000.0
        live_latencies.append(elapsed_ms)

        detections = []
        boxes = np.empty((0, 7), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        if pred_corners is not None:
            corners_np = pred_corners.detach().cpu().numpy()
            boxes = box_utils.corner_to_center(corners_np, order="lwh")
            scores = pred_scores.detach().cpu().numpy()
            for box, score in zip(boxes, scores):
                detections.append({
                    "label": "vehicle",
                    "score": float(score),
                    "box": [float(value) for value in box],
                })

        frame_result = {
            "frame_index": index,
            "agent_id": "ego",
            "coordinate_frame": "ego_lidar",
            "model_name": "pointpillars",
            "inference_ms": elapsed_ms,
            "detections": detections,
        }
        frames.append(frame_result)
        print("frame={} detections={} time_ms={:.1f}".format(
            index, len(detections), elapsed_ms))

        origin_lidar = lidar_to_numpy(batch_data["ego"]["origin_lidar"], np)
        if live is not None:
            plt, mpl_transforms, Rectangle, figure, axis = live
            if not plt.fignum_exists(figure.number):
                print("Live window closed; stopping inference.")
                break
            axis.clear()
            axis.set_facecolor("#111827")
            axis.scatter(origin_lidar[::2, 0], origin_lidar[::2, 1], s=0.45,
                         c=origin_lidar[::2, 2], cmap="viridis", alpha=0.75)
            for box, score in zip(boxes, scores):
                x, y, _, length, width, _, yaw = box
                patch = Rectangle((x - length / 2, y - width / 2),
                                  length, width, fill=False,
                                  edgecolor="#ff4d4f", linewidth=2.5)
                patch.set_transform(mpl_transforms.Affine2D().rotate_around(
                    x, y, yaw) + axis.transData)
                axis.add_patch(patch)
                axis.text(x, y + width, "{:.2f}".format(score),
                          color="#ff4d4f", fontsize=10, ha="center")
            axis.scatter([0], [0], marker="^", s=150, color="#40a9ff",
                         label="Ego")
            lidar_range = hypes["preprocess"]["cav_lidar_range"]
            axis.set_xlim(lidar_range[0], lidar_range[3])
            axis.set_ylim(lidar_range[1], lidar_range[4])
            axis.set_aspect("equal")
            axis.grid(alpha=0.15)
            axis.legend(loc="upper right")
            axis.set_xlabel("Forward / x (m)")
            axis.set_ylabel("Left / y (m)")
            stable_values = live_latencies[1:] if len(live_latencies) > 1 else live_latencies
            average_ms = sum(stable_values) / len(stable_values)
            axis.set_title(
                "LIVE Ego-only PointPillars | Frame {} | {} boxes | {:.1f} ms | avg {:.1f} ms".format(
                    index, len(detections), elapsed_ms, average_ms),
                fontsize=15, weight="bold")
            figure.tight_layout()
            figure.canvas.draw_idle()
            plt.pause(max(0.001, args.live_delay))

        if args.save_vis:
            dataset.visualize_result(
                pred_corners, gt_corners, batch_data["ego"]["origin_lidar"],
                False, str(vis_dir / "{:05d}.png".format(index)), dataset=dataset)
        if args.save_npy:
            np.save(str(args.output_dir / "frame_{:05d}_pcd.npy".format(index)),
                    origin_lidar)

    output_path = args.output_dir / "detections.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump({"mode": "strict_ego_only", "frames": frames}, stream,
                  ensure_ascii=False, indent=2)
    print("Saved: {}".format(output_path))
    if live is not None and live[0].fignum_exists(live[3].number):
        print("Inference finished. Close the live window to exit.")
        live[0].ioff()
        live[0].show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
