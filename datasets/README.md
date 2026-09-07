# 数据集说明（Task 9 固定附件）

面向 `patrol_perception` 工程：无自建数据时，用公开集冷启动；自采数据用于补域差与稀有事件。

设计依据：`docs/superpowers/specs/2026-08-05-patrol-perception-yolo26-design.md`（含 **§2.0 客户标配 AI 事件预警**）  
实施计划（事件可/不可实现 + 公开集分波 W0–W6）：`docs/superpowers/plans/2026-08-05-patrol-perception-yolo26.md`  
训练脚本落地后见：`scripts/train_*.sh`、`config/dataset_*.yaml`  
数据源开关与链接：`config/dataset_sources.yaml`  
类名与四类起步：`config/classes.yaml`  
**下载/映射可复制命令**：`datasets/DOWNLOAD_COMMANDS.md`


---

## 0. FAQ：下载、多余类、平衡、会不会忘掉 COCO

### 0.1 网上集要不要下到本地？

**要。** Ultralytics 微调读的是本地 `images/` + `labels/`（或本地 `data.yaml` 指向的路径）。  
流程：`url` → 下载到 `dataset_sources.yaml` 的 `local_dir`（如 `datasets/downloads/...`）→ **清洗/映射** → 合并进 `datasets/detect_v1/` → `yolo detect train`。

### 0.1b 数据放到外置硬盘（绝对路径）

可以。推荐三种方式（优先级从高到低）：

1. **CLI**：`--data-root /mnt/your_hdd/patrol_data`
2. **环境变量**：`export PATROL_DATA_ROOT=/mnt/your_hdd/patrol_data`
3. **yaml**：在 `roots.data_root` 写绝对路径

相对路径会拼成：`data_root + local_dir`，例如：

```text
data_root: /mnt/your_hdd/patrol_data
local_dir: datasets/downloads/coco
→ /mnt/your_hdd/patrol_data/datasets/downloads/coco
```

某一条也可直接写绝对 `local_dir` / `self.dir`（不再拼接 data_root）。  
配置里继续保留相对路径即可，换机器只改 `data_root`。

```bash
cd patrol_perception
export PATROL_DATA_ROOT=/mnt/your_hdd/patrol_data
PYTHONPATH=src python datasets/scripts/download_from_sources.py --dry-run
# 或
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --data-root /mnt/your_hdd/patrol_data --classes person,bicycle,car,truck
```

W0 **直接用 `yolo26s.pt` 推理**即可。联合微调时：底座类 `person/bicycle/car/truck` 的 COCO `public` 已 `enabled: true`，下载脚本按 `url`+`local_dir` **去重只下一份**（默认 **val2017** → `datasets/downloads/coco_val2017`）。

```bash
cd patrol_perception
PYTHONPATH=src python datasets/scripts/download_from_sources.py --dry-run
PYTHONPATH=src python datasets/scripts/download_from_sources.py \
  --classes person,bicycle,car,truck --coco-variant val2017 \
  --data-root /media/sribd/C02E01372E01284C/patrol_perception
# 冒烟子集：--coco-variant coco128
# 完整可复制命令见 datasets/DOWNLOAD_COMMANDS.md
```

### 0.2 下载的集和需要的类不一致怎么办？（例：只要井盖，却带了坑洞、路损…）

公开集很少「刚好只有你的一个类」。标准处理是 **类映射 + 过滤**，不是原样整包塞进训练：

```text
下载原始包 (downloads/xxx)
    → 读官方 names / 标注
    → class_map：只保留需要的源类 → 本工程 class_id
       例：Open Manhole / uncovered manhole → manhole_hazard
           Pothole / Sewer Cover → 丢弃（或暂不写入 labels）
    → 写出新 labels/*.txt（仅保留映射后的行）
    → 若某图过滤后没有任何框：可删图，或保留作负样本（空 txt）
    → 再并入 detect_v1
```

映射表放 `config/class_maps/`（如 `manhole_roboflow.yaml`）。  
**没用的类 = 不进最终 labels**，磁盘上可留着原始包，训练不读它们的框。

### 0.3 数据和标签怎么「平衡」？

常见三类不平衡：

| 问题 | 做法 |
| --- | --- |
| 某类图特别多（如 COCO person） | 联合训时对该集 **降采样** / 限制每 epoch 最大图数 |
| 某类特别少（如 fishing_rod） | **过采样**、复制、更强增广；或提高该类 loss 权重 |
| 一图多类/一集多类 | 先按上节 **只映射目标类**；负样本（无目标图）适量保留压误报 |
| 多数据集合并 | 按类统计框数，调各集 `fraction` 或采样权重，避免某一集主导梯度 |

验收看各类 **AP**，不要只看总 mAP。稀有类 AP 掉、COCO 旧类也掉 → 要调采样或回放（见下）。

### 0.4 在 `yolo26s.pt` 上「一批批」换新数据训，会不会忘掉原来的能力？

**会，这就是灾难性遗忘（catastrophic forgetting）。**  
若只在烟火集上训很多 epoch，再只在井盖集上训，模型容易弱化 person/car/bicycle 等原 COCO 能力。

**正确做法（与你的理解一致）：微调新类时，把 COCO（或 COCO 子集）与新数据一起训。**

```text
yolo26s.pt
  + 新类数据（smoke/ebike/manhole/… 映射过滤后）
  + COCO 回放（至少 person/bicycle/car/truck 等 always_include）
  → 联合 data.yaml → detect_best.pt
```

实践要点：

1. **要下 COCO 到本地**（或 Ultralytics 自动拉取的 coco 子集），才能和新政据一起读进训练。  
2. **不必每次下全量 118k**：常用 **COCO 子集 / 按类抽样**（例如每类数千图，或官方 `coco128`/`coco8` 冒烟 + 更大抽样）作回放即可；全量 COCO 更稳但磁盘与时间成本高。  
3. `events_catalog.yaml` → `train.always_include_detect`（person、bicycle、car、truck…）表示联合训时这些类必须有回放样本。  
4. 底座 COCO 条目已默认 `enabled: true`；与新类 public/self **同一轮合并训练**。多类共用同一 `local_dir` 时下载自动去重。  
5. 分波 W1→W6 只做验证；**最终交付权重**用「新类 ∪ COCO 回放」联合训出，避免纯串行覆盖。

**不推荐：** 只用新集、不带回放，从上一波权重继续训。  
**推荐：** 始终 `新数据 + COCO 回放` 联合（可分期加大新类比例，但旧类不能为 0）。

---

## 0b. 与客户标配事件的数据优先级

一期**必须覆盖**产品「AI 事件预警」标配（设计文档 §2.0）。数据准备建议顺序：

| 优先级 | 事件/类 | 数据策略 |
| --- | --- | --- |
| A（先做） | 自行车、小轿车、大货车、猫、狗、水瓶 | **预训练直接用**；域差大再自制微调 |
| B | 电动车、烟雾、明火、四类垃圾、吸烟、井盖等 | 公开专项集（映射过滤）+ 自采混训 |
| C | 聚集、躺卧、夜间滞留 | 检测框 + **规则**（少标行为） |
| D | 钓鱼、翻越、斗殴、人脸 | 公开行为/杆/脸集 + 规则；牵绳仍偏自采 |
| E | 噪音 | **音频**，不进 YOLO 图集 |

---

## 1. 本仓库数据布局约定

Ultralytics YOLO 检测推荐布局：

```text
patrol_perception/datasets/detect_v0/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/          # 与图像同名的 .txt
│   └── val/
└── data.yaml           # path / train / val / names
```

分割任务同理（标签为 YOLO seg 多边形格式）。  
原始下载包放 `datasets/downloads/`（已 gitignore），转换结果放 `datasets/<name>/`。

标签格式（检测，每行一个目标）：

```text
class_id x_center y_center width height
```

均为相对图像宽高的 **0–1 归一化**坐标。

---

## 2. 推荐公开数据集清单（按类）

> 链接与托管页可能变更；下载前核对官方页。许可以官方为准，下表「许可」为常见摘要，**商用必须自行复核**。

### 2.1 检测类

| 本工程类名 | 推荐公开集 | 获取方式 | 许可摘要 | 冷启动价值 | 域差/风险 |
| --- | --- | --- | --- | --- | --- |
| `person`, `vehicle`, 部分骑行相关 | [COCO](https://cocodataset.org/) | 官网下载；或 Ultralytics 训练时用内置 `coco.yaml` / 小集 `coco8.yaml` | 研究常用；商用看条款 | **最高**（底座） | 视角非巡逻车头 |
| `person` / `car` / `truck` / `bus` / `bike` / `traffic light` 等 | [BDD100K](https://bdd-data.berkeley.edu/) | 官网申请下载 | 研究协议为主 | 高（更偏道路） | 驾驶视角 ≠ 低矮巡逻视角 |
| `traffic_light`（框） | COCO / BDD100K | 同上 | 同上 | 高 | 状态（红绿）需另做分类 |
| `cone_or_barrier` | Roboflow Universe 搜 `traffic cone` / `road barrier`；部分 BDD 相关标注 | Roboflow 导出 YOLOv8/YOLO 格式 | 各数据集不同 | 中高 | 质量参差，选 star 高、协议清晰的 |
| `license_plate` | [CCPD](https://github.com/detectRecog/CCPD)（中国车牌检测常用）；CRPD；Roboflow 车牌检测集 | GitHub/网盘说明；注意地区车牌差异 | 多为研究 | 高（检框） | 号牌**识别**用 OCR，不必 YOLO 读字符 |
| `litter` | [TACO](http://tacodataset.org/)；UAVVaste；Roboflow `street litter` | TACO 官网/GitHub | TACO 等需看 LICENSE | 中（大件垃圾） | 碎屑/烟头仍难；类要合并映射 |
| `cigarette_or_smoke` | Roboflow / Kaggle：`smoking`、`cigarette detection` | 平台导出 YOLO | 多为研究/非商用 | 低–中（M1） | 小目标、域差大，必须自采补 |
| `cyclist` / `scooter` | COCO 骑行相关 + Roboflow `scooter` / `motorcycle` | 导出后合并类 | 各异 | 中 | 与 `person`+`bicycle` 规则可二选一 |
| `animal` | COCO（dog/cat/…）合并为 `animal` 或保留细类 | COCO | 同上 | 中 | 巡逻场景动物少，可后加 |
| `trash_bin` | Roboflow `trash bin` / `garbage can` | 导出 YOLO | 各异 | 中 | 满溢状态需分类或规则 |
| `gate_or_door` | Roboflow 门/闸机相关集 | 导出 YOLO | 各异 | 低–中 | 场地相关，优先自采 |

### 2.2 分割类

| 本工程类名 | 推荐公开集 | 获取方式 | 许可摘要 | 说明 |
| --- | --- | --- | --- | --- |
| `water_puddle` | Roboflow / 论文附集搜 `puddle`、`water hazard` | 优先选已提供 YOLO-seg 或可转 polygon 的 | 各异 | 定义「多大算积水」要在 `classes.yaml` 写死 |
| `pothole` / `road_damage` | 道路坑洞检测/分割公开集（Roboflow、部分挑战赛） | 导出 YOLO-seg 或先框后转 | 各异 | 可先检测框，再升 seg |

### 2.3 姿态 / 行为（可不先训）

| 需求 | 推荐做法 | 说明 |
| --- | --- | --- |
| 跌倒启发式 | 直接用 Ultralytics **`yolo26n-pose.pt` / `yolo26s-pose.pt` 预训练** | 一期可零标注 |
| 跌倒专用微调 | 各 Fall Detection 图/视频集（来源杂） | 监控俯视多，与车载侧视差大 |
| 斗殴 | **几乎无可靠工业公开框集** | 用多人靠近 + pose 规则；自采/摆拍补 M1→M2 |

### 2.4 信号灯状态分类

| 需求 | 推荐做法 |
| --- | --- |
| 红 / 黄 / 绿 / other | 从 COCO/BDD/`traffic_light` 检测框裁剪 ROI，人工或半自动标 4 类；或找 `traffic light classification` 公开集 |

### 2.5 OCR（车牌号）

不进入 YOLO 训练头。推理阶段对 `license_plate` 裁剪后用 PaddleOCR / EasyOCR 等。公开车牌图可作 OCR 联调，不必改检测标签格式。

---

## 3. 下载方式（实操要点）

### 3.1 COCO（推荐最先跑通）

```bash
# 环境
source /home/sribd/D1_EDU/Pytorch/switch_cuda_mode.sh 1

# 冒烟：Ultralytics 自动拉 coco8
yolo detect train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=1 batch=2 \
  project=/home/sribd/D1_EDU/patrol_perception/data/runs/detect name=coco8_smoke
```

完整 COCO：按 [COCO 下载页](https://cocodataset.org/#download) 取 train/val 图像与 annotations，再用第 4 节工具转 YOLO，或直接使用 Ultralytics 提供的 `coco.yaml`（会按文档指引准备数据）。

### 3.2 Roboflow Universe

1. 打开对应项目页 → Download → 选 **YOLOv8** / **Ultralytics YOLO** 格式  
2. 解压到 `datasets/downloads/<name>/`  
3. 按第 5 节做 **类名重映射** 后拷贝进 `datasets/detect_v0/`

### 3.3 CCPD 等专用集

按仓库 README 下载；多数为检测框（或需从四角点转 AABB）。转成 YOLO txt 后，类 id 固定为你方 `license_plate` 的 id。

### 3.4 TACO 等 COCO-JSON 标注集

下载图像 + COCO 格式 `annotations.json`，用第 4 节脚本转为 YOLO，并把 TACO 细类合并映射到单一 `litter`（或保留子类再在规则层合并）。

---

## 4. 如何转成 YOLO 格式

### 4.1 已是 YOLO txt（Roboflow 导出）

只需：

1. 统一目录为 `images/{train,val}` + `labels/{train,val}`  
2. 编写 `data.yaml` 的 `names` 与本工程 `config/classes.yaml` **顺序一致**（或做 id 重映射）  
3. 训练：`yolo detect train model=yolo26s.pt data=datasets/detect_v0/data.yaml ...`

### 4.2 COCO JSON → YOLO txt

可用 Ultralytics 自带转换思路，或官方/社区脚本。示意逻辑：

```text
对每个 annotation:
  category_id → 映射为本工程 class_id（见下节）
  bbox [x,y,w,h]（COCO 为左上角 + 宽高，像素）
  → x_c = (x + w/2) / img_w
  → y_c = (y + h/2) / img_h
  → bw = w / img_w
  → bh = h / img_h
  写入 labels/<split>/<stem>.txt 一行
```

实现脚本计划放：`scripts/coco_to_yolo.py`（Task 9 编码时落地）。在此之前可用：

```bash
# 示例：部分环境可用
pip install ultralytics
# 按 ultralytics 文档使用 converter，或临时用 roboflow 在线转
```

### 4.3 分割：COCO polygon / mask → YOLO-seg

YOLO-seg 每行：`class_id x1 y1 x2 y2 ...`（归一化多边形顶点）。  
有 mask 无 polygon 时，需轮廓提取后再归一化。Roboflow 导出 YOLO-seg 可减少手写转换。

### 4.4 分类（信号灯）

目录型 ImageFolder：

```text
datasets/tl_cls_v0/
├── train/red|yellow|green|other/
└── val/red|yellow|green|other/
```

`yolo classify train model=yolo26n-cls.pt data=datasets/tl_cls_v0 ...`

---

## 5. 类名重映射（混多源必做）

多源合并时维护一张表（建议另存 `datasets/class_map_public.yaml`）：

```yaml
# 示例：公开集原始名 → 本工程名
person: person
car: vehicle
truck: vehicle
bus: vehicle
motorcycle: cyclist   # 或单独类，按 classes.yaml 定
traffic light: traffic_light
cigarette: cigarette_or_smoke
smoking: cigarette_or_smoke
```

规则：

1. 本工程 `config/classes.yaml` 的 `detect_names` 顺序 = 训练 `data.yaml` 的 id 顺序  
2. 丢弃本工程不需要的公开类（或单独实验）  
3. 负样本：可保留无目标图（空 txt）抑制误报  

---

## 6. 推荐冷启动配方（无自采时）

| 阶段 | 数据 | 目标 |
| --- | --- | --- |
| A | `coco8` / 小 COCO 子集 | 环境与训练脚本冒烟 |
| B | COCO 人车相关类 + 锥桶/车牌/垃圾公开子集 | `detect_v0`，离线管线真权重 |
| C | 自采巡逻视角 10%–30% 混入 | 降域差，冲 M2 |
| D | 吸烟/积水/斗殴/禁停区专项补采 | 稀有事件 |

公开:自采 常见从 `9:1` → `7:3`。验证集尽量含 **Gemini336 / 实车视角** 图片，否则指标虚高。

---

## 7. 不要指望公开集单独搞定的项

- 场地「违规停车」语义（靠围栏规则，不是公开「违停」类）  
- 稳定吸烟、斗殴告警  
- 与你们相机高度、畸变、夜间完全一致的表现  

这些在成熟度上看板标 **M1**，用自采升级到 M2/M3。

---

## 8. 与实施计划的关系

- 本文件 = **Task 9 固定附件**，类清单与下载/转换流程以本文为准迭代  
- 转换脚本、`dataset_detect.example.yaml`、`train_detect.sh` 在 Task 9 编码任务中创建，路径均在 `patrol_perception/` 下  
- 大数据与权重不进 git（见仓库根 `.gitignore`）
