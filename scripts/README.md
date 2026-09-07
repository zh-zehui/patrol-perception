# scripts/ 运行说明

本目录为 **训练 / 评测 / 权重冒烟** 入口脚本。数据合并、预览等见 [`../datasets/DOWNLOAD_COMMANDS.md`](../datasets/DOWNLOAD_COMMANDS.md) 与 `../datasets/scripts/`。

工作目录一律为工程根：

```bash
cd /home/sribd/D1_EDU/patrol_perception
```

## 共用环境

| 项 | 说明 |
| --- | --- |
| conda | `pytorch_cu130` |
| CUDA | 必须先 `source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1`，否则 `import torch` / ultralytics 可能缺库失败 |
| 数据根 | `PATROL_DATA_ROOT` 或 `DATA_ROOT`，默认外置盘 `/media/sribd/C02E01372E01284C/patrol_perception` |

```bash
source /home/sribd/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_cu130
source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1
cd /home/sribd/D1_EDU/patrol_perception
```

---

## 1. `train_detect_w1_fire.sh` — W1 烟火检测训练

**作用：** 以 `yolo26l.pt` 为起点，在联合集 `detect_w1_fire` 上微调；结束后把 `best.pt` 拷到 `data/weights/detect_w1.pt`；默认再跑一遍 `eval_detect_run.py`。

**前置：**

- 已生成 `$DATA_ROOT/datasets/detect_w1_fire/data.yaml`（见 merge / `merge_job`）
- 存在 `data/weights/yolo26l.pt`

### 基本用法

```bash
./scripts/train_detect_w1_fire.sh
```

只训练、跳过训后评测：

```bash
EVAL_AFTER=0 ./scripts/train_detect_w1_fire.sh
```

### 环境变量

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `PATROL_DATA_ROOT` | 外置盘路径 | 数据根（内含 `datasets/detect_w1_fire`） |
| `DATA_YAML` | `$PATROL_DATA_ROOT/datasets/detect_w1_fire/data.yaml` | 训练 data.yaml |
| `WEIGHTS` | `data/weights/yolo26l.pt` | 起点权重 |
| `RUN_NAME` | `w1_fire` | Ultralytics run 名 → `data/runs/detect/<RUN_NAME>/` |
| `EPOCHS` | `50` | 轮数 |
| `BATCH` | `16` | batch |
| `EVAL_AFTER` | `1` | `1`=训后评测，`0`=跳过 |
| `EVAL_BASELINE` | （空） | 训后传 `--baseline`；空则用 `eval.yaml` 的 `baseline.default` |
| `EVAL_NO_BASELINE` | `0` | `1`=训后评测不加遗忘基线（传 `--no-baseline`，优先于 `EVAL_BASELINE`） |
| `WAVE` | `w1_fire` | 训后 `--wave`，须与 `config/eval.yaml` 的 `focus_classes_by_wave` 键一致；联合集用 `w2_joint` |

示例：改轮数与 run 名

```bash
EPOCHS=30 RUN_NAME=w1_fire_e30 ./scripts/train_detect_w1_fire.sh
```

联合 15 类重训并按 W2 focus 出报告：

```bash
WAVE=w2_joint RUN_NAME=w2_joint ./scripts/train_detect_w1_fire.sh
```

单独评测已有权重：

```bash
PYTHONPATH=src python scripts/eval_detect_run.py \
  --weights data/weights/detect_w1.pt \
  --data "$PATROL_DATA_ROOT/datasets/detect_w1_fire/data.yaml" \
  --wave w2_joint \
  --config config/eval.yaml
```

### 产物路径

| 路径 | 说明 |
| --- | --- |
| `data/runs/detect/<RUN_NAME>/weights/best.pt` | 验证集最优权重（Ultralytics 按 `project`+`name` 写入） |
| `data/runs/detect/<RUN_NAME>/weights/last.pt` | 最后一轮权重（续训用） |
| `data/weights/detect_w1.pt` | 脚本从 `best.pt` **复制**的稳定别名，供推理/评测引用 |
| `data/eval_runs/<wave>_<timestamp>/` | `EVAL_AFTER=1` 时由评测脚本生成 |

训练保存目录由命令参数决定：

```text
project=data/runs/detect  name=$RUN_NAME
→ data/runs/detect/$RUN_NAME/weights/{best,last}.pt
```

---

## 2. `eval_detect_run.py` — 训后 / 独立离线评测

**作用：** 在 **val（必跑）** 与 **test（data.yaml 有则跑）** 上算 AP 等指标；可选相对 baseline 的 base 类 ΔAP（遗忘检查）。这是**带标注的离线评测**，不是摄像头在线推理。

**配置：** `config/eval.yaml`（`splits`、`focus_classes_by_wave`、`base_classes`、`baseline.default`、输出目录）。

直接跑本脚本时仍可用 `--baseline` 覆盖 YAML。`train_detect_w1_fire.sh` 默认不传 `--baseline`（走 YAML）；覆盖用 `EVAL_BASELINE=/path/to.pt`，跳过用 `EVAL_NO_BASELINE=1`。

### 基本用法（与训练脚本调用一致）

```bash
PYTHONPATH=src python scripts/eval_detect_run.py \
  --weights data/weights/detect_w1.pt \
  --data "$PATROL_DATA_ROOT/datasets/detect_w1_fire/data.yaml" \
  --wave w1_fire \
  --config config/eval.yaml
```

也可用训练 run 里的 `best.pt`：

```bash
--weights data/runs/detect/w1_fire/weights/best.pt
```

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--weights` | 待评权重 `.pt`（必填） |
| `--data` | 联合集 `data.yaml`（必填） |
| `--wave` | 波次键，须与 `eval.yaml` 里 `focus_classes_by_wave` 匹配（如 `w1_fire`） |
| `--baseline` | 覆盖 `eval.yaml` 的 `baseline.default`；只在 **val** 上算 base 类 Δ |
| `--no-baseline` | 不做基线 val / ΔAP |
| `--splits` | 覆盖默认，如 `val` 或 `val,test` |
| `--max-images N` | 每 split 等间隔抽 N 张做快速评测（大 val 冒烟） |
| `--imgsz` / `--batch` / `--device` | 传给 Ultralytics `val` |
| `--dry-metrics-json` | 不跑 GPU，用假指标测报告落盘 |

### 大 val 快速评测

```bash
PYTHONPATH=src python scripts/eval_detect_run.py \
  --weights data/weights/detect_w1.pt \
  --data "$PATROL_DATA_ROOT/datasets/detect_w1_fire/data.yaml" \
  --wave w1_fire \
  --baseline data/weights/yolo26l.pt \
  --max-images 400
```

说明：Ultralytics 的 `fraction` **只对 train 生效**，对 val/test 无效；要用 `--max-images`。

### 产物

```text
data/eval_runs/<wave>_<timestamp>/
  report.md                 # 含 Split: val 与（若有）Split: test
  metrics_val.json
  metrics_test.json         # 可选
  metrics_baseline_val.json # 有 baseline 时（YAML 默认或 --baseline）
  baseline_delta.json
  run_meta.json
data/eval_runs/index.jsonl  # 追加一行索引
```

`report.md` 中 **val** 用于调参/遗忘对照；**test** 同为离线 AP，不是部署推理评估。

---

## 3. `smoke_yolo26_weights.py` — YOLO26 预训练权重冒烟

**作用：** 加载 `yolo26{n,s,m,l,x}.pt`，在样例图上跑一次 detect，确认 GPU/权重可用。

```bash
python scripts/smoke_yolo26_weights.py --sizes n,s
# 或全部规格
python scripts/smoke_yolo26_weights.py --sizes n,s,m,l,x
```

预览图写在 `data/weights/smoke_*.jpg`。

---

## 推荐流程（W1）

```text
1. 准备/更新 detect_w1_fire（merge / merge_job）
2. ./scripts/train_detect_w1_fire.sh
      → best.pt + detect_w1.pt + 默认 eval report
3. 需要单独复评时再跑 eval_detect_run.py
```

数据集脚本（merge、preview、summarize）不在本目录，见：

- [`../datasets/DOWNLOAD_COMMANDS.md`](../datasets/DOWNLOAD_COMMANDS.md)
- [`../datasets/README.md`](../datasets/README.md)
