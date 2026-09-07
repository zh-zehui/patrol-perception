# patrol_perception

巡逻车视觉感知独立工程（YOLO26：告警 L1 / 巡检 L2 / 导航辅助 L3）。

本目录与 `d1_perception`（激光代价地图）、`d1_navigation`、`Pytorch/` **分离**，仅通过约定接口与外部环境协作：

| 外部依赖 | 用法 |
| --- | --- |
| Docker `d1-sim` + Gemini336 | 录包 / 出图（不在本仓库内改驱动） |
| 主机 conda `pytorch_cu130` | 训练与离线推理 |
| 设计文档 | `docs/superpowers/specs/2026-08-05-patrol-perception-yolo26-design.md` |
| 实施计划 | `docs/superpowers/plans/2026-08-05-patrol-perception-yolo26.md` |

## 目录结构（规划）

```text
patrol_perception/
├── README.md                 # 本文件
├── datasets/                 # 数据集说明、公开集清单、转换笔记
│   └── README.md             # Task 9 固定附件：公开数据集清单
├── config/                   # classes / events / infer / zones（待 Task 1）
├── schema/                   # FrameResult JSON Schema（待 Task 1）
├── src/patrol_perception/    # Python 包（待实现）
├── scripts/                  # 训练 / 抽帧 / 演示脚本
├── tests/
├── ros_bridge/               # 二期 ROS2 节点
└── data/                     # 本地大数据（gitignore）：raw / weights / runs …
```

## 快速入口

- **公开数据冷启动**：见 [`datasets/README.md`](datasets/README.md)
- **训练 / 评测脚本用法**：见 [`scripts/README.md`](scripts/README.md)（`train_detect_w1_fire.sh`、`eval_detect_run.py`、冒烟）
- **环境**：`source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1`，使用 `pytorch_cu130`
- **许可**：Ultralytics YOLO 为 AGPL-3.0；公开数据集各有许可，商用前务必核对

## 环境已就绪（2026-08-05）

| 项 | 值 |
| --- | --- |
| conda | `pytorch_cu130` |
| torch | **2.10.0**（未改动；安装 ultralytics 时用了 `--no-deps`） |
| ultralytics | **8.4.115**（≥8.3.0，满足 YOLO26） |
| CUDA | 可用（`torch.cuda.is_available() == True`） |
| 预训练权重 | `data/weights/yolo26{n,s,m,l,x}.pt` |

安装要点（复现）：

```bash
source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1
pip install -U "ultralytics>=8.3.0" --no-deps   # 避免覆盖 Jetson 专用 torch
# 再装 ultralytics 运行依赖（勿升级 torch）：
pip install pyyaml opencv-python-headless requests matplotlib scipy pandas seaborn psutil polars "ultralytics-thop>=2.0.18" "nvidia-ml-py>=12.0.0"
```

冒烟测试（GPU，bus 图）：

```bash
cd /home/sribd/D1_EDU/patrol_perception
source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1
python scripts/smoke_yolo26_weights.py --sizes n,s   # 或 n,s,m,l,x
```

实测：`yolo26n/s/m/l/x` 均加载并推理成功；预览图见 `data/weights/smoke_bus_yolo26n.jpg`、`smoke_bus_yolo26s.jpg`。工程默认后续训练/部署用 **yolo26s**。

## 训后评测（Detect Eval）

完整参数与产物说明见 **[`scripts/README.md`](scripts/README.md)**。下面为最短示例。

每波微调（W1 烟火、…）训完后，用统一 CLI 在 val（+test）上出标准指标，并可选对基线权重算遗忘相关的 ΔAP。详见设计文档 [`docs/superpowers/specs/2026-08-11-detect-eval-framework-design.md`](../docs/superpowers/specs/2026-08-11-detect-eval-framework-design.md)。

**环境（必需）**：与训练/推理一致，先激活 CUDA 模式，否则 `import ultralytics/torch` 会报 `ImportError: libnvpl_lapack...`：

```bash
source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1
cd /home/sribd/D1_EDU/patrol_perception
```

**真实评测**（val 必跑；data.yaml 含 test 则自动追加）：

```bash
PYTHONPATH=src python scripts/eval_detect_run.py \
  --weights data/weights/detect_w1.pt \
  --data /path/to/detect_w1_fire/data.yaml \
  --wave w1_fire \
  --config config/eval.yaml
```

**大 val 集冒烟/快速评测**：Ultralytics 的 `fraction` 仅对 `train` 生效，对 `val`/`test` 是 no-op；集内 val 达数千张图时用 `--max-images N` 做等间隔抽样子集（临时 `.txt` 图片列表，不改动源数据集）：

```bash
PYTHONPATH=src python scripts/eval_detect_run.py \
  --weights data/weights/yolo26l.pt \
  --data /path/to/detect_w1_fire/data.yaml \
  --wave w1_fire_smoke_cli \
  --baseline data/weights/yolo26l.pt \
  --max-images 400
```

**无 GPU 冒烟**（只测编排/落盘/index，不跑推理）：`--dry-metrics-json path/to/fake_metrics.json`（见 `scripts/eval_detect_run.py --help` 中的 JSON 结构说明）。

产物：`data/eval_runs/<wave>_<timestamp>/{run_meta.json, metrics_val.json, metrics_test.json, metrics_baseline_val.json, baseline_delta.json, report.md}`，并追加一行到 `data/eval_runs/index.jsonl`。

## 状态

环境与 YOLO26 基础权重冒烟已完成；业务代码按实施计划 Task 1 起实现；检测评测框架（`eval/` + `scripts/eval_detect_run.py`）已完成并冒烟通过。
