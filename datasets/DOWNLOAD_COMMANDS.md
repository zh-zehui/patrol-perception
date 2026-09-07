# 公开数据下载命令速查

面向 `datasets/scripts/download_from_sources.py`：按 `config/dataset_sources.yaml` 里 **enabled public** 任务去重下载。  
类映射表在 `config/class_maps/`；设计总览见 `datasets/README.md`。

---

## 0. 每次先做

```bash
cd /home/sribd/D1_EDU/patrol_perception
# conda activate pytorch_cu130   # 如需

export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception
# 也可：export PATROL_DATA_ROOT=$DATA_ROOT
```

通用参数：


| 参数                                               | 含义                                                            |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `--data-root $DATA_ROOT`                         | 外置盘数据根（推荐）                                                    |
| `--classes a,b,c`                                | 只跑这些类的 enabled public；省略则跑 yaml 全部 enabled                    |
| `--force`                                        | 忽略 `.download_ok.json` 强制重下                                   |
| `--dry-run`                                      | 只打印去重后的任务，不下载                                                 |
| `--coco-variant val2017|train2017|coco128|coco8` | COCO 变体；**默认 val2017**；`train2017` 约 18GB，落到 `coco_train2017` |
| `--ndjson /path/to/x.ndjson`                     | 覆盖 yaml 里的 ndjson 路径                                          |
| `--ndjson-concurrency 8`                         | ndjson 拉图并发（默认 8）                                             |
| `--taco-workers 16`                              | TACO 并行线程                                                     |
| `--taco-full`                                    | TACO 下原图（更大更慢）；默认 640 图                                       |


先看会下什么：

```bash
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --data-root "$DATA_ROOT" --dry-run
```

---

## 0b. 长时间任务用 tmux（推荐：COCO train2017 / 训练）

`tmux` 是终端复用器：在机器上开一个**可脱离**的会话。关掉 Cursor 终端、SSH 断开后，里面的下载/训练**继续跑**；之后再 `attach` 回来看进度。

### 常用命令一览


| 命令                                    | 作用                            |
| ------------------------------------- | ----------------------------- |
| `which tmux`                          | 确认已安装（本机一般在 `/usr/bin/tmux`）  |
| `tmux ls`                             | 列出当前所有会话；没有会话时可能无输出           |
| `tmux new -s coco`                    | **前台**新建名为 `coco` 的会话并进入      |
| `tmux new-session -d -s coco …`       | **后台**新建会话并立刻跑命令（不占当前终端）      |
| `tmux attach -t coco`                 | 连回名为 `coco` 的会话看实时输出          |
| `tmux capture-pane -t coco -p -S -40` | 不进入会话，只打印最近约 40 行输出（适合脚本偷看进度） |
| `tmux kill-session -t coco`           | 结束该会话（**会杀掉**里面正在跑的进程）        |
| 在会话内：`Ctrl+b` 再按 `d`                  | **detach**：离开会话，任务继续          |


前缀键默认是 `Ctrl+b`（先松开，再按后面的字母）。

### 下载前检查磁盘与目录

```bash
# 外置盘空间（train2017 zip ~18GB，解压后更大，建议预留 ≥40GB）
df -h /media/sribd/C02E01372E01284C | tail -2

# 目标目录是否已存在
ls /media/sribd/C02E01372E01284C/patrol_perception/datasets/downloads/coco_train2017 \
  2>/dev/null || echo 'coco_train2017 not yet'
```

### 推荐：后台一键开会话并开始下 train2017

在**普通终端**执行（不要先手动 `tmux attach`）：

```bash
# 若已有同名会话且想重来：先结束旧会话（会停掉旧下载）
tmux has-session -t coco 2>/dev/null && tmux kill-session -t coco

tmux new-session -d -s coco -c /home/sribd/D1_EDU/patrol_perception bash -lc '
source /home/sribd/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_cu130
export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception
echo "[tmux coco] start $(date)"
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes person,bicycle,car,truck \
  --data-root "$DATA_ROOT" \
  --coco-variant train2017
echo "[tmux coco] done $(date) exit=$?"
exec bash
'
```

逐段含义：


| 片段                                     | 含义                                               |
| -------------------------------------- | ------------------------------------------------ |
| `tmux has-session -t coco`             | 是否已有名为 `coco` 的会话                                |
| `&& tmux kill-session -t coco`         | 有则先杀掉，避免重名冲突                                     |
| `new-session -d`                       | 新建并立刻 **detach**（后台跑）                            |
| `-s coco`                              | 会话名，以后用这个名字 attach                               |
| `-c …/patrol_perception`               | 会话起始工作目录                                         |
| `bash -lc '…'`                         | 登录式 bash 跑引号里整段脚本（能读 conda）                      |
| `source …/conda.sh` + `conda activate` | 非交互环境也能 `activate`（与 `switch_cuda_mode.sh` 同类问题） |
| `exec bash`                            | 下载结束后留在 shell 里，方便 attach 后看最后日志；不会因命令结束而关掉会话    |


确认已起来：

```bash
sleep 2
tmux ls
# 期望类似：coco: 1 windows (created …)

# 偷看输出（不进入）
tmux capture-pane -t coco -p -S -30
```

正常开头会看到 `local_dir=…/coco_val2017 → …/coco_train2017`，随后先下 YOLO labels，再下 `train2017.zip`（进度条约 `x/18.0GB`）。

### 日常查看 / 离开

```bash
# 进入会话看实时进度
tmux attach -t coco

# 看够了：Ctrl+b，松开，再按 d  → 回到普通终端，下载不停

# 另开一个窗口只看目录增长（不必 attach）
ls -lh "$DATA_ROOT/datasets/downloads/coco_train2017"
# 下载中常见：labels/ 已就绪 + 逐渐变大的 train2017.zip
```

### 下完之后

会话里出现 `[tmux coco] done … exit=0`，且目录有 `.download_ok.json`、`data.yaml`、`images/train2017/`。然后：

```bash
python datasets/scripts/summarize_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/downloads/coco_train2017" \
  --classes person,bicycle,car,truck
```

不再需要该会话时可：`tmux kill-session -t coco`。

### 交互式写法（等价、步骤更多）

```bash
tmux new -s coco
# 进入后手动：
source /home/sribd/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_cu130
cd /home/sribd/D1_EDU/patrol_perception
export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes person,bicycle,car,truck \
  --data-root "$DATA_ROOT" \
  --coco-variant train2017
# 然后 Ctrl+b d 离开
```

训练脚本同样建议放进 tmux（会话名可改成 `w1train` 等），用法相同。

---

## 1. COCO 防遗忘回放

### 1a. val2017（约 5k 图，已有则 skip）

底座类 `person/bicycle/car/truck` 共用一份，落到 `datasets/downloads/coco_val2017`。

```bash
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes person,bicycle,car,truck \
  --data-root "$DATA_ROOT" \
  --coco-variant val2017
```

### 1b. train2017（约 118k 图 / ~18GB，推荐补 bicycle 不平衡）

yaml 里 `local_dir` 仍写 `coco_val2017` 时，选 `train2017` 会**自动改落到** `datasets/downloads/coco_train2017`，不覆盖 val。

**请用 tmux 跑**（命令解释见上方 [§0b](#0b-长时间任务用-tmux推荐coco-train2017--训练)）；直接在 Cursor 终端跑，关窗口容易中断。

```bash
# 后台会话 coco 一键启动（完整解释见 §0b）
tmux has-session -t coco 2>/dev/null && tmux kill-session -t coco
tmux new-session -d -s coco -c /home/sribd/D1_EDU/patrol_perception bash -lc '
source /home/sribd/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_cu130
export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes person,bicycle,car,truck \
  --data-root "$DATA_ROOT" \
  --coco-variant train2017
exec bash
'
tmux attach -t coco   # 查看；离开：Ctrl+b 然后 d
```

```bash
# 统计（下完后）
python datasets/scripts/summarize_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/downloads/coco_train2017" \
  --classes person,bicycle,car,truck
```

合并 W1（train2017 进 train + val2017 整池进 val）见 `config/merge_w1_fire.yaml`，或：

```bash
bash datasets/scripts/download_w1_fire.sh "$DATA_ROOT"
# 仅用 val 回放：COCO_VARIANT=val2017 bash datasets/scripts/download_w1_fire.sh
```

冒烟子集（旧目录 `datasets/downloads/coco`，仅 128 张）：

```bash
# 若 yaml 仍指向 coco_val2017，冒烟请临时改 local_dir 或单独放：
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes person \
  --data-root "$DATA_ROOT" \
  --coco-variant coco128 \
  --force
```

> 补底座不平衡优先 **train2017**；仅冒烟/快速回放可用 **val2017**；`coco128` 只够跑通链路。

---

## 2. 烟火（HuggingFace fog_fire）

`smoke` / `flame` 共用同一 HF 仓，去重只下一份。

```bash
# 需 HF token：patrol_perception/.hf_token 或 export HF_TOKEN=hf_xxx
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes smoke,flame \
  --data-root "$DATA_ROOT"
```

映射 Fire/Smoke → `flame`/`smoke`：

```bash
python datasets/scripts/remap_yolo_set.py \
  --src "$DATA_ROOT/datasets/downloads/fog_fire_detection/dataset" \
  --out "$DATA_ROOT/datasets/downloads/fog_fire_yolo" \
  --map config/class_maps/fire_smoke.yaml
```

---

## 3. ndjson（Platform 导出 → 本地 YOLO）

先在 [Ultralytics Platform](https://platform.ultralytics.com/) 下载 `.ndjson` 到 `~/Downloads/`（或任意路径，用 `--ndjson` 指定）。

### 3.1 电瓶车 / 街景（ebike）

```bash
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes ebike \
  --data-root "$DATA_ROOT" \
  --ndjson ~/Downloads/datasets.ndjson \
  --ndjson-concurrency 8
```

```bash
# 将 datasets-<hash> 换成实际目录名
python datasets/scripts/remap_yolo_set.py \
  --src "$DATA_ROOT/datasets/downloads/ebike_hub/datasets-27bb5609" \
  --out "$DATA_ROOT/datasets/downloads/ebike_yolo" \
  --map config/class_maps/ebike_campus.yaml
```

扁平纯电瓶车图（无标注、白/黑底商品图）先转 YOLO，再增量追加：

```bash
python datasets/scripts/flat_images_to_yolo.py \
  --src "$DATA_ROOT/datasets/downloads/e-bike" \
  --out "$DATA_ROOT/datasets/downloads/e-bike_yolo" \
  --class-name ebike

python datasets/scripts/merge_yolo_sets.py --mode append \
  --config config/merge_w2_ebike2.yaml \
  --out "$DATA_ROOT/datasets/detect_w1_fire"
```

### 3.2 香烟（cigarette）

```bash
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes cigarette_or_smoke \
  --data-root "$DATA_ROOT" \
  --ndjson ~/Downloads/cigarette.ndjson \
  --ndjson-concurrency 8
```

```bash
# 将 cigarette-<hash> 换成实际目录名
python datasets/scripts/remap_yolo_set.py \
  --src "$DATA_ROOT/datasets/downloads/cigarette_roboflow/cigarette-<hash>" \
  --out "$DATA_ROOT/datasets/downloads/cigarette_yolo" \
  --map config/class_maps/cigarette_face.yaml
```

### 3.3 钓鱼（fishing）

```bash
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes fishing_rod \
  --data-root "$DATA_ROOT" \
  --ndjson ~/Downloads/fishing.ndjson \
  --ndjson-concurrency 8
```

```bash
python datasets/scripts/remap_yolo_set.py \
  --src "$DATA_ROOT/datasets/downloads/fishing_recognition/fishing-34d02471" \
  --out "$DATA_ROOT/datasets/downloads/fishing_yolo" \
  --map config/class_maps/fishing_rod.yaml
```

### 3.4 稳健转换（可选，可续跑）

与上面 `--classes … kind=ndjson` 等价，中断后可直接再跑：

```bash
PYTHONUNBUFFERED=1 python datasets/scripts/convert_ndjson_resilient.py \
  --ndjson ~/Downloads/fishing.ndjson \
  --out "$DATA_ROOT/datasets/downloads/fishing_recognition" \
  --concurrency 8 --timeout 180 --max-rounds 10
```

---

## 4. TACO 垃圾

`litter_bottle` / `litter_plastic_bag` / `litter_paper` / `litter_cardboard` 共用一份 TACO。

```bash
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes litter_bottle,litter_plastic_bag,litter_paper,litter_cardboard \
  --data-root "$DATA_ROOT" \
  --taco-workers 16
```

COCO JSON → YOLO（工程 litter 类）：

```bash
python datasets/scripts/taco_to_yolo.py \
  --taco-root "$DATA_ROOT/datasets/downloads/taco" \
  --out "$DATA_ROOT/datasets/downloads/taco_yolo" \
  --map config/class_maps/taco_litter.yaml
```

---

## 5. 一次跑完所有 enabled public

```bash
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --data-root "$DATA_ROOT" \
  --coco-variant val2017 \
  --ndjson-concurrency 8 \
  --taco-workers 16
```

注意：

- **ndjson** 类需本机已有对应 `.ndjson`（路径见 yaml 的 `ndjson:` 或用 `--ndjson` 覆盖；一次 CLI 只能覆盖一个 ndjson，多集请按类分次跑）。
- **kind: page**（仅备案搜索页 / 需人工导出）会打印 manual 提示，不会自动拉完。
- 已成功的任务有 `.download_ok.json`，再跑会 `[skip]`；要重下加 `--force`。

---

## 6. 多集合并（Task 3：`merge_yolo_sets.py`）

将已 remap 的子集按全局 `names` 合并（默认软链图片，前缀防重名）。

```bash
cd /home/sribd/D1_EDU/patrol_perception
export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception

# W1 烟火 + COCO → detect_w1_fire
# 烟火、COCO 均：源图合成一池后按 split_ratios 默认 7:2:1 → train/val/test
python datasets/scripts/merge_yolo_sets.py \
  --config config/merge_w1_fire.yaml \
  --out "$DATA_ROOT/datasets/detect_w1_fire"

# 增量追加（--mode append）：保留已有 train/val/test，只并入本 config 里的新源；
# names 必须以现有 data.yaml 为前缀，只在末尾加新类；碰撞同名文件会报错。
# W2 示例（ebike / TACO / fishing / cigarette / COCO cat·dog）：
python datasets/scripts/merge_yolo_sets.py --mode append \
  --config config/merge_w2_append.yaml \
  --out "$DATA_ROOT/datasets/detect_w1_fire"

# 或一键：下载(若缺) + remap + merge（默认 COCO_VARIANT=train2017）
bash datasets/scripts/download_w1_fire.sh "$DATA_ROOT"
# 仅 val 回放：COCO_VARIANT=val2017 bash datasets/scripts/download_w1_fire.sh "$DATA_ROOT"
```

CLI 拼装（无 config 文件时）：

```bash
python datasets/scripts/merge_yolo_sets.py \
  --out "$DATA_ROOT/datasets/detect_w1_fire" \
  --names person,bicycle,car,truck,smoke,flame \
  --set "$DATA_ROOT/datasets/downloads/fog_fire_yolo:fire:flame,smoke" \
  --set "$DATA_ROOT/datasets/downloads/coco_train2017:coco_tr:person,bicycle,car,truck" \
  --set "$DATA_ROOT/datasets/downloads/coco_val2017:coco_va:person,bicycle,car,truck"
```

`--mode`：


| 值          | 行为                                               |
| ---------- | ------------------------------------------------ |
| `full`（默认） | 清空 `--out` 后按 config 全量重建                        |
| `append`   | 不删旧图/旧标签；只写入本 config 的 `sets`；扩展 `names` 只能追加在末尾 |
| `remove`   | 按 `sets.prefix` 或 `--prefix` 删除已写入文件；不改 `names`  |


```bash
# 按 config 里的 prefix 剔除（如 merge_w2_ebike2.yaml 的 ebike2）
python datasets/scripts/merge_yolo_sets.py --mode remove \
  --config config/merge_w2_ebike2.yaml \
  --out "$DATA_ROOT/datasets/detect_w1_fire"

# 或直接指定前缀
python datasets/scripts/merge_yolo_sets.py --mode remove \
  --prefix ebike2 \
  --out "$DATA_ROOT/datasets/detect_w1_fire"
```

登记 job 一键 remap + append / remove（`merge_job.sh`，登记表 `config/merge_jobs.yaml`：
`fire`、`coco`、`coco_pet`、`taco`、`fishing`、`cig`、`ebike`、`bicycle2`、`fishing2`、`cigarette2`、`ebike2`）：

```bash
cd /home/sribd/D1_EDU/patrol_perception
export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception

# 列出 prefix、candidates、present/absent 状态
bash datasets/scripts/merge_job.sh list

# W1 coco：只保留 person（方案 B：整包删后按 remain 重并）
bash datasets/scripts/merge_job.sh remove coco --drop bicycle,car,truck
# 或两步：整包删再 append
# bash datasets/scripts/merge_job.sh remove coco -y
# bash datasets/scripts/merge_job.sh append coco --keep person

# 首次并入：COCO cat+dog remap 为 pet
bash datasets/scripts/merge_job.sh append coco_pet

# 非交互：全部 candidates keep（即 pet）
bash datasets/scripts/merge_job.sh append coco_pet -y

# 显式 keep
bash datasets/scripts/merge_job.sh append coco_pet --keep pet

# 整包删除且不重 append
bash datasets/scripts/merge_job.sh remove coco_pet -y

# 预览临时 merge yaml / remove 计划，不执行
bash datasets/scripts/merge_job.sh append coco_pet --dry-run
bash datasets/scripts/merge_job.sh remove coco --drop bicycle,car,truck --dry-run

# 后期 job 示例（remap + 按 7:2:1 append）
bash datasets/scripts/merge_job.sh append bicycle2
bash datasets/scripts/merge_job.sh append pet2 -y

# 未登记的新集：旧接口，先准备 class_maps/*.yaml 与 merge_w2_*.yaml
bash datasets/scripts/merge_job.sh append \
  --src "$DATA_ROOT/datasets/downloads/Foo.v1i.yolo26" \
  --map config/class_maps/foo.yaml \
  --remap-out "$DATA_ROOT/datasets/downloads/foo_yolo" \
  --config config/merge_w2_foo.yaml

bash datasets/scripts/merge_job.sh remove --prefix foo2
```

已 present 的 prefix 再 `append` 会 `[skip]` 跳过；改 keep 子集用 `remove`（可 `--drop`）后再 `append`（或 `--drop` 一次完成）。`remove` 后全局 `names` **不收缩**（去掉的类 id 仍保留，可能暂时无框）。底座 `fire`/`coco` 与 `merge_w1_fire.yaml` 同源；重建整库仍用 full merge，日常用本入口做 prefix 级增删。

---

## 7. 查看已下载 / 合并集类别分布

```bash
# COCO 只看四类底座
python datasets/scripts/summarize_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/downloads/coco_val2017" \
  --classes person,bicycle,car,truck

# W1 联合集（train/val/test 全拆）
python datasets/scripts/summarize_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire"

# 可选落盘 JSON
python datasets/scripts/summarize_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire" \
  --json /tmp/w1_class_stats.json
```

## `stats_yolo_pos_neg.py`：正/负样本与按类统计

统计单个 YOLO 检测集的图级正负样本、各类 boxes/图数，以及按 split 分布。  
适用于 `images/{train,val,test}` + `labels/...`，也支持扁平 `images/` + `labels/`（报告里 split 名为 `all`）。  
根目录需有 `data.yaml`（读 `names`）。

### 判定规则（按图）


| 情况                           | 计数                        |
| ---------------------------- | ------------------------- |
| 有同 stem 的 `.txt`，且 strip 后非空 | **pos**（正样本）              |
| 有 `.txt`，但内容为空               | **neg**（负样本 / 空标签）        |
| 没有对应 `.txt`                  | **missing_label**（不算 neg） |


正负互斥。`per-class` 只在 **pos** 上解析 YOLO 行（`class_id cx cy w h`）。

### 参数


| 参数         | 说明                              |
| ---------- | ------------------------------- |
| `--root`   | YOLO 集根（必填，含 `data.yaml`）       |
| `--rename` | 报告用类名映射，`src:dst,...`（**不改磁盘**） |
| `--json`   | 可选，把完整结果写成 JSON                 |


### 输出字段（控制台）

- **overall**：`images / pos / neg / missing_label`，`neg:pos`，正样本上 boxes 合计与均值，多类共现图数  
- **splits**：各 split 的 imgs/pos/neg/miss  
- **per-class**：`boxes`（框数）、`%box`（占全部框）、`imgs`（至少有一个该类框的图数）、`%pos`（占全部正样本图）  
- **class images by split**：各类有框图数按 train/val/test

### 示例

```bash
cd /home/sribd/D1_EDU/patrol_perception
export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception

# 实采 litter 源集（带源类名 → 工程类名，仅报告）
python datasets/scripts/stats_yolo_pos_neg.py \
  --root "$DATA_ROOT/datasets/downloads/02_trash_full_yolo_5271_images" \
  --rename bottle:litter_bottle,box:litter_cardboard,paper:litter_paper,bag:litter_plastic_bag \
  --json /tmp/trash_stats.json

# 单类 lying 集
python datasets/scripts/stats_yolo_pos_neg.py \
  --root "$DATA_ROOT/datasets/downloads/yolo_dataset_export_20260813/03_lying_yolo_265_images"

# 联合集（若几乎无空标签，neg 会接近 0）
python datasets/scripts/stats_yolo_pos_neg.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire" \
  --json /tmp/joint_pos_neg.json
```

与 `summarize_yolo_dataset.py` 的区别：后者偏各类 boxes/%img；本脚本额外拆清 **pos / neg / missing**，并给出 `neg:pos` 与按 split 的类图数。

```bash
# 实采 litter → 工程类名后 append（默认丢空标签负样本；合并时 7:2:1）
# bash datasets/scripts/merge_job.sh append trash2 -y
# 若已按旧划分并过：先 remove trash2 -y 再 append，才会按 7:2:1 重划
# 负样本更推荐用下方 add_negatives_to_joint.py（不必改 trash2）
```

## `add_negatives_to_joint.py`：向联合集注入 / 删除空标签负样本

把空标签图以固定 **prefix**（默认 `bg`）append 进联合集（默认 `$DATA_ROOT/datasets/detect_w1_fire`）。  
空标签对 **全部类** 同时为负样本。流程：收集候选 → 抽样 → 写 staging → `merge_yolo_sets --mode append --keep-empty`。

### 流程要点

1. **候选来源**（可混用、可重复）：
  - `--from-yolo ROOT`：扫该 YOLO 的 train/val/test（或扁平 `images/`），只收 **空标签 / 缺 txt** 图；**打乱原 split**，混成一池  
  - `--from-images DIR`：递归收集图片，全部当负样本（staging 时写空 txt）
2. **正样本基数 `P`**：来自联合集 `--out`（默认 `detect_w1_fire`）的 train+val+test **有框图总数**，**不是** `--from-yolo` 里的正样本数
3. **目标数量**
  `T = min(候选数 C, 可选 ceil(P × ratio), 可选 max)`；都不给则用尽候选
4. **划分**：对抽中的 `T` 张按 `--split-ratios`（默认 `0.7 0.2 0.1`）**重新**划 train/val/test，再 append
5. **prefix**：写入文件名为 `<prefix>_...`；已存在同 prefix 须加 `--replace`（先删再加）

### 参数


| 参数               | 默认                                                            | 说明                                          |
| ---------------- | ------------------------------------------------------------- | ------------------------------------------- |
| `--out`          | `$DATA_ROOT/datasets/detect_w1_fire`                          | 联合集根                                        |
| `--data-root`    | 环境变量 `DATA_ROOT`                                              | 用于推导默认 `--out` / `--staging`                |
| `--from-yolo`    | —                                                             | 可重复；YOLO 空标签源                               |
| `--from-images`  | —                                                             | 可重复；未标注图片目录                                 |
| `--ratio`        | —                                                             | 相对联合集正样本图数 `P`                              |
| `--max`          | —                                                             | 上限张数                                        |
| `--seed`         | `0`                                                           | 抽样与划分                                       |
| `--split-ratios` | `0.7 0.2 0.1`                                                 | train/val/test                              |
| `--prefix`       | `bg`                                                          | 合并前缀                                        |
| `--staging`      | `$DATA_ROOT/datasets/downloads/joint_negatives_<prefix>_yolo` | 临时 YOLO 包                                   |
| `--replace`      | off                                                           | 已有同 prefix 时先 remove 再 append               |
| `--remove`       | off                                                           | **只删除**该 prefix，不注入；不可与源路径 / `--replace` 同用 |
| `--dry-run`      | off                                                           | 只打印 JSON 摘要，不写盘                             |


`--staging` 不得与 `--out` 或任一输入源路径重叠（相等或互为祖先），否则 exit 2，避免误删数据。

### dry-run / 结果字段

```text
{"positives": P, "candidates": C, "target": T, "split_counts": {"train": …, "val": …, "test": …}}
```

例：`P=132510, C=3266, ratio=0.3, max=5000` → `T = min(3266, ceil(39753), 5000) = 3266`（受候选上限）。

### 示例

```bash
cd /home/sribd/D1_EDU/patrol_perception
export DATA_ROOT=/media/sribd/C02E01372E01284C/patrol_perception

# 1) 预览（不写盘）
python datasets/scripts/add_negatives_to_joint.py \
  --from-yolo "$DATA_ROOT/datasets/downloads/02_trash_full_yolo_5271_images" \
  --ratio 0.3 --max 5000 --dry-run

# 2) 写入（YOLO 空标签 + 可选扁平背景图）；已有 bg 时加 --replace
python datasets/scripts/add_negatives_to_joint.py \
  --from-yolo "$DATA_ROOT/datasets/downloads/02_trash_full_yolo_5271_images" \
  --from-images /path/to/patrol_bg_frames \
  --ratio 0.3 --max 5000 --seed 0 --replace

# 3) 自定义划分与 prefix
python datasets/scripts/add_negatives_to_joint.py \
  --from-yolo "$DATA_ROOT/datasets/downloads/02_trash_full_yolo_5271_images" \
  --split-ratios 0.8 0.1 0.1 --prefix bg_trash --replace

# 4) 只删除已注入的负样本包（默认 prefix=bg）
python datasets/scripts/add_negatives_to_joint.py --remove
python datasets/scripts/add_negatives_to_joint.py --remove --prefix bg --dry-run

# 5) 注入后核对联合集空标签是否上升
python datasets/scripts/stats_yolo_pos_neg.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire" \
  --json /tmp/joint_after_bg.json
```

按类 / prefix 抽图拼图预览（每组两张：`*_raw.jpg` 无框 + `*_labeled.jpg` 带框）。
`--mode class` 按类别抽；省略 `--prefixes` 时按 `merge_meta.json` 交互选源前缀（如 `cigarette_or_smoke` → cig / cig2 / all）。

```bash
# 给类、不给 prefix → 按 merge_meta 提示选（cig / cig2 / all）
python datasets/scripts/preview_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire" \
  --out /tmp/preview_cig \
  --classes cigarette_or_smoke --mode class

# 直接指定 prefix
python datasets/scripts/preview_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire" \
  --out /tmp/preview_cig2 \
  --classes cigarette_or_smoke --prefixes cig2 --mode class

# 跳过提示，相关 prefix 全选
python datasets/scripts/preview_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire" \
  --out /tmp/preview_cig_all \
  --classes cigarette_or_smoke --mode class -y

# 一次多类 / 多源（非交互）
python datasets/scripts/preview_yolo_dataset.py \
  --root "$DATA_ROOT/datasets/detect_w1_fire" \
  --out /tmp/detect_w1_preview_subset \
  --split val --per-group 9 \
  --classes ebike,litter_bottle,fishing_rod,cigarette_or_smoke \
  --prefixes taco,cig2,bike2,ebike \
  --mode class
```

输出目录含 `by_class/`（及可选 `by_prefix/`）、`index.md`、`preview_meta.json`。


| 内容                | 路径（相对 `$DATA_ROOT`）                                         |
| ----------------- | ----------------------------------------------------------- |
| COCO val2017 回放   | `datasets/downloads/coco_val2017`                           |
| COCO train2017 回放 | `datasets/downloads/coco_train2017`                         |
| COCO128 冒烟（旧）     | `datasets/downloads/coco`                                   |
| 烟火原始 / 映射后        | `.../fog_fire_detection` → `.../fog_fire_yolo`              |
| ebike 原始 / 映射后    | `.../ebike_hub/datasets-*` → `.../ebike_yolo`               |
| 香烟 原始 / 映射后       | `.../cigarette_roboflow/cigarette-*` → `.../cigarette_yolo` |
| 钓鱼 原始 / 映射后       | `.../fishing_recognition/fishing-*` → `.../fishing_yolo`    |
| TACO 原始 / 映射后     | `.../taco` → `.../taco_yolo`                                |
| 合并训练根（规划）         | `datasets/detect_v1`                                        |
| W1 烟火联合集          | `datasets/detect_w1_fire`                                   |


开关与 URL 以 `config/dataset_sources.yaml` 为准；改 `enabled` / `ndjson` 路径后重新跑对应 `--classes` 即可。