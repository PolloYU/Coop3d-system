"""Create a simple weekly-meeting BEV report from single-3D results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, transforms
from matplotlib.patches import Rectangle
import numpy as np


def configure_chinese_font():
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            font = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.sans-serif"] = [font.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def demo_data():
    rng = np.random.RandomState(42)
    points = np.column_stack((rng.uniform(-35, 35, 5000),
                              rng.uniform(-20, 20, 5000),
                              rng.normal(-1.2, 0.25, 5000)))
    boxes = [
        {"label": "vehicle", "score": 0.94,
         "box": [8.0, 2.5, -0.4, 4.2, 1.8, 1.6, 0.08]},
        {"label": "vehicle", "score": 0.87,
         "box": [20.0, -5.5, -0.4, 4.5, 2.0, 1.7, -0.18]},
        {"label": "vehicle", "score": 0.76,
         "box": [-12.0, 7.0, -0.4, 3.9, 1.7, 1.5, 0.42]},
    ]
    for item in boxes:
        x, y, _, length, width, _, yaw = item["box"]
        local = rng.normal(size=(300, 2)) * [length / 3, width / 3]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]])
        rotated = local.dot(rotation.T)
        cluster = np.column_stack((rotated[:, 0] + x, rotated[:, 1] + y,
                                   rng.uniform(-0.8, 0.8, len(local))))
        points = np.vstack((points, cluster))
    return points, {"mode": "synthetic_demo", "frames": [{
        "frame_index": 0, "agent_id": "ego", "coordinate_frame": "ego_lidar",
        "model_name": "pointpillars", "inference_ms": 28.4,
        "detections": boxes,
    }]}


def load_results(result_path, pcd_path):
    with result_path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    points = np.load(str(pcd_path)) if pcd_path and pcd_path.is_file() else None
    return points, result


def draw_box(axis, box, score):
    x, y, _, length, width, _, yaw = box
    patch = Rectangle((x - length / 2, y - width / 2), length, width,
                      fill=False, edgecolor="#ff4d4f", linewidth=2)
    patch.set_transform(transforms.Affine2D().rotate_around(x, y, yaw) + axis.transData)
    axis.add_patch(patch)
    axis.text(x, y + width, "{:.2f}".format(score), color="#ff4d4f",
              fontsize=8, ha="center")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    parser.add_argument("--pcd", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/weekly_demo/single_3d_report.png"))
    parser.add_argument("--frame-index", type=int, default=0,
                        help="Frame position inside detections.json")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    chinese = configure_chinese_font()

    if args.demo:
        points, result = demo_data()
    elif args.results and args.results.is_file():
        points, result = load_results(args.results, args.pcd)
    else:
        raise FileNotFoundError("Provide --results or use --demo")

    frames = result.get("frames", [])
    if not frames:
        raise ValueError("No frames found in result JSON")
    if args.frame_index < 0 or args.frame_index >= len(frames):
        raise IndexError("frame-index {} outside [0, {})".format(
            args.frame_index, len(frames)))
    frame = frames[args.frame_index]
    detections = frame.get("detections", [])

    fig = plt.figure(figsize=(13, 6.5), facecolor="#f5f7fa")
    grid = fig.add_gridspec(1, 2, width_ratios=(2.2, 1), wspace=0.08)
    bev = fig.add_subplot(grid[0, 0])
    info = fig.add_subplot(grid[0, 1])
    bev.set_facecolor("#111827")
    if points is not None and len(points):
        bev.scatter(points[:, 0], points[:, 1], s=0.35, c=points[:, 2],
                    cmap="viridis", alpha=0.65)
    for detection in detections:
        draw_box(bev, detection["box"], detection["score"])
    bev.scatter([0], [0], marker="^", s=120, color="#40a9ff", label="Ego")
    box_x = [item["box"][0] for item in detections]
    box_y = [item["box"][1] for item in detections]
    x_extent = max([40.0] + [abs(value) + 7.0 for value in box_x])
    y_extent = max([25.0] + [abs(value) + 5.0 for value in box_y])
    bev.set_xlim(-x_extent, x_extent)
    bev.set_ylim(-y_extent, y_extent)
    bev.set_aspect("equal")
    bev.set_xlabel("Forward / x (m)")
    bev.set_ylabel("Left / y (m)")
    bev.set_title("Ego-only PointPillars | BEV")
    bev.grid(alpha=0.12)
    bev.legend(loc="upper right")

    info.axis("off")
    real = result.get("mode") != "synthetic_demo"
    status = (("真实推理结果" if real else "模拟版式预览") if chinese
              else ("REAL INFERENCE" if real else "SYNTHETIC PREVIEW"))
    info.text(0.02, 0.94, status,
              fontsize=18, weight="bold", color="#389e0d" if real else "#d48806")
    if chinese:
        model_name = frame.get("model_name", "pointpillars")
        if str(model_name).lower() == "pointpillars":
            model_name = "PointPillars"
        rows = [("输入", "一帧 Ego LiDAR 点云"),
                ("模型", model_name),
                ("输出", "{} 个车辆 3D 检测框".format(len(detections))),
                ("单帧耗时", "{:.1f} ms".format(frame.get("inference_ms") or 0.0))]
    else:
        rows = [("Input", "One ego LiDAR point cloud"),
                ("Model", frame.get("model_name", "pointpillars")),
                ("Output", "{} vehicle boxes".format(len(detections))),
                ("Latency", "{:.1f} ms / frame".format(frame.get("inference_ms") or 0.0))]
    y = 0.84
    for label, value in rows:
        info.text(0.02, y, label, fontsize=12, weight="bold")
        info.text(0.02, y - 0.05, value, fontsize=11)
        y -= 0.14
    info.text(0.02, 0.30, "Pipeline",
              fontsize=12, weight="bold")
    pipeline = ("PCD → Voxelization → Pillar features\n→ BEV backbone → 3D boxes"
                if chinese else
                "PCD -> voxelization -> pillars\n-> BEV backbone -> 3D boxes")
    info.text(0.02, 0.21, pipeline,
              fontsize=10.5, linespacing=1.5)
    if not real:
        note = ("仅用于展示版式，不是模型结果。" if chinese
                else "Layout demo only; not a model result.")
        info.text(0.02, 0.07, note,
                  fontsize=10, color="#ad6800")
    title = ("Coop3D 组内周报 | 单端 3D 检测" if chinese
             else "Coop3D Weekly Demo | Single-agent 3D Detection")
    fig.suptitle(title,
                 fontsize=17, weight="bold", y=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
