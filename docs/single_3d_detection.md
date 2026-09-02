# 单端 3D 检测模块

## 我们负责什么

输入一辆车自己的激光雷达点云，输出该车坐标系下的车辆等目标 3D 检测框。输出框统一为：

`[x, y, z, length, width, height, yaw] + label + score`

这一步是协同感知的基础：后续融合组会接收每个车/路侧节点的检测结果或中间特征。单端组暂时不负责多车坐标变换、网络通信模拟和跨节点融合。

## 第一版技术路线

- 数据：先使用 OPV2V 的 ego 车辆点云和标注。
- 模型：PointPillars，先复现 OpenCOOD 提供的基线。
- 评估：3D AP、推理耗时，并按正常/遮挡场景分别统计。
- 接口：`models/single_detection/contracts.py` 固定检测结果格式。

## 当前代码怎么用

先检查本机环境：

```powershell
python tools/check_single_3d_environment.py
```

在 OpenCOOD 的 Python 3.7环境中，不依赖数据和权重构建一次 PointPillars：

```bash
python tools/smoke_pointpillars.py --opencood-root /mnt/d/shixi/OpenCOOD
```

数据下载并解压后，先检查目录和点云/标注是否配对：

```powershell
python tools/inspect_opv2v.py D:\path\OPV2V
```

准备好 OpenCOOD、模型目录和数据后，先只检查并打印命令：

```powershell
python tools/run_single_detection.py --opencood-root D:\path\OpenCOOD --model-dir D:\path\checkpoint
```

确认无误后增加 `--execute` 执行 OpenCOOD 推理。

严格 ego 单端小规模推理（默认只处理10帧）：

```bash
python tools/infer_single_opv2v.py \
  --opencood-root /mnt/d/shixi/OpenCOOD \
  --model-dir /mnt/d/shixi/checkpoints/pointpillar_late_fusion \
  --data-root /mnt/d/shixi/datasets/OPV2V/test \
  --output-dir /mnt/d/shixi/Coop3d-system/outputs/single_3d \
  --max-frames 10 --save-vis --save-npy
```

该入口运行前会强制 `max_cav=1`，并逐场景确认唯一输入节点就是 ego；输出统一写入 `detections.json`，可视化写入 `vis/`。

周会演示图：

```powershell
# 数据未就绪时，只生成明确标注的示意图
python tools/visualize_single_3d_report.py --demo

# 真实推理完成后
python tools/visualize_single_3d_report.py `
  --results outputs/single_3d/detections.json `
  --pcd outputs/single_3d/frame_00000_pcd.npy
```

## 当前边界

OpenCOOD 官方推理入口按数据集和融合模式运行。第一阶段用 `late` 模式验证安装、数据和权重链路；严格单端基线需要在数据适配器中让每帧只保留 ego 节点。不能把 late fusion 的多节点结果当作单端成绩。

单端实验必须满足以下规则：模型输入只能包含当前 ego 节点的点云；训练、验证、测试使用固定且互不混用的划分；指标在同一坐标范围、类别和 IoU 阈值下比较。以后即使把 OpenCOOD 配置设成 `max_cav: 1`，也必须在样本加载处确认留下的节点确实是 ego，不能只看配置名称。

真实推理还需要：兼容的 OpenCOOD 环境、OPV2V 数据集、PointPillars 预训练权重。模型训练开始前，还要核对数据划分、类别、坐标系和框定义。
