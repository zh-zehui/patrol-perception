# D1 数据集标注系统

本工具用于现有 **YOLO Detection** 数据集的本地可视化审核和人工修正。它不会修改类别名称或类别 ID；界面颜色仅供区分，不会写入数据集。

## 1. 第一版能力

- 直接读取 `images/**` 与对应的 `labels/**.txt`；
- 浏览器显示图片、类别和检测框；
- 拖动框改变位置，拖动四角改变大小；
- 新建、删除、改类、撤销和重做；
- 上一张/下一张、文件名筛选、快捷键；
- 标记“已审核”或“需复核”；
- 保存标准 YOLO：`class_id x_center y_center width height`；
- 覆盖前自动把旧标签备份到数据集的 `.d1_annotations/history/`；
- 使用临时文件 + `os.replace` 原子保存，降低中断导致标签损坏的风险；
- 默认只监听 `127.0.0.1`，数据和图片不上传外部服务。

当前第一版是**标注编辑器**。YOLO-World/Grounding DINO 属于下一层“预标注适配器”，建议放在另一个 GPU 环境中生成候选 YOLO TXT，再用本工具审核。这样不会让基础编辑器依赖 PyTorch/CUDA，也不会污染现有 `pytorch_cu130` 环境。

## 2. 固定类别（不可改序）

```text
0 person                 7 litter_bottle
1 bicycle                8 litter_plastic_bag
2 car                    9 litter_paper
3 truck                 10 litter_cardboard
4 smoke                 11 fishing_rod
5 flame                 12 cigarette_or_smoke
6 ebike                  13 pet
```

语义约定：

- `pet` 包含猫和狗（含无法从图片判断归属的流浪猫狗），每只单独框选；
- `cigarette_or_smoke` 沿用已有名称，在本系统中表示**香烟实体**；
- `smoke` 表示明显的非吸烟环境烟雾；
- 类别配置位于 `config/classes.yaml`，自动标注可使用多个内部英文 prompt，但导出始终回到上述名称和 ID。

## 3. 独立 Miniforge 环境

在本目录执行：

```bash
cd /home/sribd/D1_EDU/patrol_perception/annotation_tool
/home/sribd/miniforge3/bin/conda env create -f environment.yml
conda activate d1-autolabel
```

如果环境已经创建，更新命令是：

```bash
/home/sribd/miniforge3/bin/conda env update -n d1-autolabel -f environment.yml --prune
```

## 4. 数据目录

推荐标准结构：

```text
/path/to/my_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

图片可以暂时没有同名 TXT，保存后系统会创建标签目录。请先备份重要数据；工具虽保存历史版本，但 `.d1_annotations/` 不代替正式数据版本管理。

## 5. 启动

```bash
conda activate d1-autolabel
d1-annotate --dataset /path/to/my_dataset
```

浏览器访问：<http://127.0.0.1:8765>。

在局域网另一台电脑访问时才使用：

```bash
d1-annotate --dataset /path/to/my_dataset --host 0.0.0.0 --port 8765
```

此版本没有账号系统，只能在可信局域网使用；不要直接暴露到互联网。

## 6. 常用操作

| 操作 | 方法 |
|---|---|
| 选中/移动框 | 单击并拖动框内部 |
| 调整大小 | 拖动选中框的四角 |
| 新建框 | 按 `N` 后拖动，或双击空白后拖动 |
| 改类别 | 选中框后修改右侧类别 |
| 删除框 | `Delete` |
| 保存 | `Ctrl+S` |
| 保存并审核通过 | `R` |
| 切换图片 | 左右方向键 |

## 7. 系统架构与后续自动预标注

```text
实采视频/图片
  -> 抽帧、去重、质量过滤（后续）
  -> YOLO-World 或 Grounding DINO 预标注（独立 GPU 环境，后续）
  -> 标准 YOLO TXT + 模型置信度旁车 JSON
  -> 本工具可视化审核、移动/缩放/增删框
  -> reviewed 标签版本
  -> 数据校验与 train/val/test 按采集批次切分
  -> 训练领域 YOLO，再预标注下一批数据
```

推荐环境拆分：

- `d1-autolabel`：本工具，轻量、稳定，负责审核编辑；
- `d1-autolabel-models`：后续创建，安装与设备匹配的 PyTorch、Ultralytics/YOLO-World 或 Grounding DINO；
- 不复用 ROS 环境，也不改动已有 `pytorch_cu130`。

## 8. 测试

```bash
conda activate d1-autolabel
cd /home/sribd/D1_EDU/patrol_perception/annotation_tool
source ~/.d1-autolabel-campus.env
pytest -q
```

## 9. OpenAI 视觉大模型 API 自动预标注

系统现已支持完整的“模型候选框 → 人工接受/修改 → 正式 YOLO 标签”闭环。调用的是支持图像输入的 OpenAI Responses API，并使用 Structured Outputs 限制返回结构，不是复用当前 Codex 对话会话。

### 配置

API 密钥只通过环境变量提供，不要写入仓库：

```bash
export OPENAI_API_KEY="你的 API key"
export OPENAI_VISION_MODEL="gpt-5.6"  # 可按账户实际可用的视觉模型修改

d1-annotate --dataset /path/to/my_dataset
```

未设置 `OPENAI_API_KEY` 时，人工编辑功能照常使用，大模型按钮会明确显示未配置。

### 审核流程

1. 点击“**大模型标注当前图片**”；
2. API 返回结果以**虚线候选框**显示；
3. 候选框保存在 `.d1_annotations/candidates/`，不会直接写入 `labels/`；
4. 人工点击“接受候选框”，再移动、缩放、删除、补框或改类；
5. 点击保存或“保存并通过”，才写入正式 YOLO TXT；
6. “批量标注未标注图片”使用单并发后台队列，避免瞬间并发造成限流，但仍会产生 API 费用。

安全与质量约束：

- 密钥不返回浏览器、不写日志、不保存到数据集；
- API 失败只记录异常类型，避免泄露上游请求内容；
- 模型输出不能直接成为正式训练标签；
- `cigarette_or_smoke` 仍表示香烟实体，`pet` 仍表示猫和狗；
- 通用视觉模型对香烟、远处垃圾等小目标框可能不准，必须人工审核；
- 实采图会上传到外部 API，使用前需确认隐私、授权和数据合规要求；
- API 调用会产生费用，建议先用 20～50 张代表性图片评估节省的人工时间。

官方接口依据：

- OpenAI 图像输入：<https://developers.openai.com/api/docs/guides/images-vision>
- OpenAI Structured Outputs：<https://developers.openai.com/api/docs/guides/structured-outputs>

上传前，系统会自动应用 EXIF 方向、转为 RGB JPEG，并把长边限制为 2048 像素；模型坐标使用 0～1000 归一化空间，因此仍可准确映射回原图。批量任务会按图片逐张后台处理，并在网页显示进度；不会因为数据集超过某个固定数量而拒绝任务。建议先用 20～50 张评估模型质量和 API 成本，再启动更大批次。旧版本中的 `D1_AUTOLABEL_BATCH_LIMIT` 仅为兼容保留，不再作为硬限制。

模型返回的 `confidence` 是模型自评置信度，不是经过检测数据校准的概率，只能用于人工审核排序。

### SAM 3 本地自动标注

Web 工具栏支持在“视觉大模型”和“SAM 3 本地模型”之间选择。SAM 3 运行在独立的 `d1-sam3` Miniforge 环境，通过本机 HTTP 服务返回文本提示对应的检测框、分数和提示词；两种模型都复用相同的单张/批量任务与候选审核流程。

Thor 环境安装、权重许可和启动方法见 [`sam3_service/README.md`](sam3_service/README.md)。SAM 3 的结果只保存到候选目录，不会直接修改正式 `labels/`。

## 10. Campus Sub2API 中转配置

中转声明 `wire_api = "responses"`，因此沿用 OpenAI Python SDK 的 Responses API，只需替换 `base_url` 与模型名：

```bash
conda activate d1-autolabel
export OPENAI_PROVIDER_NAME="campus_sub2api"
export OPENAI_BASE_URL="http://10.20.14.80:8080/v1"
export OPENAI_VISION_MODEL="codex.gpt-5.6-sol"
export OPENAI_REASONING_EFFORT="low"
export OPENAI_STORE="false"
export OPENAI_API_KEY="新生成的中转密钥"
export D1_AUTOLABEL_BATCH_LIMIT="50"

d1-annotate --dataset /path/to/dataset
```

也可参考 `config/campus_sub2api.env.example`。不要把真实密钥写进该示例或提交 Git。`model_catalog_json`、`personality`、`sandbox_mode` 是 Codex 客户端自身设置，不属于标注服务的 Responses API 请求参数；标注系统只使用中转地址、认证密钥、模型、推理强度和 `store=false`。

注意事项：

- 中转必须支持 Responses API 的图像 `input_image` 和 Structured Outputs；仅支持文本 Responses API 仍无法标注图片。
- 当前地址是 HTTP 明文内网地址，密钥和图片在网络链路上没有 TLS 保护，只应在可信内网使用；推荐由管理员启用 HTTPS。
- `OPENAI_STORE=false` 会向中转发送 `store: false`，但中转是否仍记录请求日志取决于中转服务自身策略，需要向管理员确认。
- 启动页只显示是否配置，不会显示 API key。

### 中转 `ValidationError` 兼容说明

0.2.1 起不再让 OpenAI SDK 严格解析中转的整个 Responses 响应信封，而是直接读取 `/v1/responses` 返回的 JSON，提取 `output_text` 后在本地校验检测结果。这兼容部分“正文正确、外层字段不完全复刻官方格式”的中转。已启动的服务不会自动加载新代码，升级后必须在服务终端按 `Ctrl+C`，重新加载 env 并启动。

若仍失败，网页会显示脱敏后的具体原因（HTTP 状态、无 `output_text`、无效 JSON或具体字段错误），可据此判断中转是否支持图像和 JSON Schema。

## 11. 忽略旧标签，由大模型独立重标

大模型请求始终只包含原始图片和固定 14 类规范，不会读取或上传已有 YOLO TXT。界面提供两种独立重标方式：

- **大模型独立标注当前图片**：为当前图片生成独立候选；
- **大模型独立重标全部图片**：无论是否已有 TXT，都为全部图片生成候选旁车文件。

候选生成后可以：

- **追加候选框**：保留已有框，用模型候选补漏；
- **候选替换旧标签**：在当前页面中丢弃旧框，以模型候选作为新框；此时尚未写入 TXT，人工修改并点击“保存并通过”后才覆盖正式标签，旧 TXT 会先进入 `.d1_annotations/history/`。

全量重标会产生较多 API 调用。现在批量任务会在后台逐张处理并显示进度，不会因为 738 张或其他数量超过旧上限而直接拒绝。推荐先独立重标 5～20 张评估质量和中转稳定性，再提高批量规模。批量任务只生成候选，不会自动覆盖正式标签；人工仍需逐图选择、调整和保存。

## 12. 图片缩放与平移

为了精细调整小目标框，标注画布支持：

- 鼠标滚轮：以指针所在位置为中心放大/缩小；
- 鼠标中键拖动：平移已放大的图片；
- 按住空格再用左键拖动：平移图片；
- “适应窗口”：恢复图片完整适应画布；
- 工具栏显示相对于适应窗口的缩放比例，范围约为 50%～2000%。

缩放和平移仅改变 Canvas 显示变换。创建、移动和缩放检测框时，系统会反算回原图归一化坐标，因此不会改变 YOLO 坐标定义或降低保存精度。

### 检测框键盘微调

选中一个正式/人工检测框后：

- `←` / `↑` / `↓` / `→`：按原图坐标移动 1 像素；
- `Shift + 方向键`：按原图坐标移动 10 像素；
- 框会被限制在图片边界内；
- 每次微调可撤销，并会标记为未保存修改。

未选中检测框时，左右方向键仍用于切换上一张/下一张图片。大模型虚线候选框需要先“追加候选框”或“候选替换旧标签”，成为正式可编辑框后才能键盘微调。

## 13. 正式旧标签、模型候选与编辑草稿分层

Web 不再把旧标签和模型候选混成一套框，当前图片包含四个相互独立的层：

1. **正式基线标签（只读）**：当前磁盘 `labels/.../*.txt`；
2. **模型候选标签（只读）**：`.d1_annotations/candidates/.../*.json`；
3. **当前编辑草稿**：人工可移动、缩放、增删和改类的内存副本；
4. **旧标签历史版本（只读）**：每次正式保存前备份在 `.d1_annotations/history/`。

操作语义：

- “候选复制并追加到草稿”：复制候选到草稿，不删除候选，不修改正式基线；
- “候选复制为新草稿”：用候选建立一份新草稿，不删除候选，不修改正式基线；
- “批量候选复制为新草稿”：对所选范围批量执行相同操作；已有草稿会先备份到
  `.d1_annotations/draft_history/` 再由候选替换，已提交图片默认跳过，正式 `labels/` 不会修改；
- “恢复正式基线到草稿”：放弃未保存草稿，从当前正式标签重新复制；
- “复制历史到草稿”：加载某个历史版本到草稿，正式基线仍不改变；
- “删除候选文件”：只删除模型候选，不影响正式基线和草稿；
- “保存/保存并通过”：唯一会将草稿提交为正式 TXT 的操作；提交前的正式 TXT 自动进入历史。

三个显示复选框可以分别隐藏/显示正式基线、模型候选和编辑草稿，便于逐层比较。正式基线使用短虚线，模型候选使用长虚线，草稿使用实线。

## 14. 批量大模型标注范围

点击“批量大模型标注”后可以选择：

1. **全部图片**：处理当前数据集所有图片；
2. **未经过保存并通过的图片**：只处理 `review.json` 中不是 `reviewed` 的图片，包含未审核和“需复核”；
3. **没有正式标签的图片**：只处理对应 `labels/.../*.txt` 不存在的图片；
4. **当前左侧筛选结果**：先在左侧按文件名筛选，再只处理筛选出的图片。

页面会先显示各范围的图片数量，再确认调用 API。批量任务仍只生成模型候选，不会直接覆盖正式标签。

“保存并通过”会在数据集的 `.d1_annotations/review.json` 中把对应图片记录为 `reviewed`。因此下一次选择“未经过保存并通过的图片”时，这些图片会自动排除；点击“需复核”的图片仍会进入该范围，便于重新处理。

### 批量候选处理策略

点击“批量大模型标注”后，除了选择图片范围，还要选择候选策略：

- **已有候选则跳过**：适合断点续跑；正式标签和已有候选都不改变；
- **重新生成候选**：重新调用模型；旧候选自动备份到 `.d1_annotations/candidate_history/`，再更新当前候选文件。

无论选择哪种策略，批量任务都不会写入或覆盖正式 `labels/**/*.txt`。正式标签只有人工把候选复制到编辑草稿、检查修改后点击“保存/保存并通过”才会更新；保存前的正式标签仍会进入 `.d1_annotations/history/`。

## 15. 多用户数据集标注门户

服务器常驻部署时，可使用一个固定门户网址管理所有已注册数据集。门户为每个数据集启动独立的标注进程和端口，因此多个用户可以同时标注不同数据集，不会共享“当前数据集”或后台任务状态。

启动：

```bash
cd /home/sribd/D1_EDU/patrol_perception/annotation_tool
./run_portal.sh
```

脚本默认加载 `~/.d1-autolabel-campus.env`，使用
`config/datasets_registry.yaml`，门户监听 `0.0.0.0:8765`。局域网用户打开：

```text
http://服务器IP:8765/
```

门户支持：

- 选择已注册数据集并进入标注；
- 注册服务器上已有的 YOLO 数据集目录；
- 首次注册后执行 `--rescan` 再进入；
- 对已注册数据集选择“重新扫描后进入”；
- 为不同数据集分配独立端口并复用正在运行的进程。

常用环境变量：

```bash
export D1_PUBLIC_HOST="服务器局域网IP或域名"
export D1_PORTAL_PORT="8765"
export D1_DATASET_REGISTRY="/path/to/datasets_registry.yaml"
export D1_AUTOLABEL_ENV_FILE="$HOME/.d1-autolabel-campus.env"
export D1_ANNOTATION_PYTHON="/home/sribd/miniforge3/envs/d1-autolabel/bin/python"
```

当前隔离粒度是“每个数据集一个进程”。不同数据集可以安全并发；同一个数据集内如果多名用户同时编辑同一张图片，仍应避免同时提交，后续可增加图片租约锁和版本冲突检测。

## 16. 批量提交管理

当前已经加入安全的“草稿 → 批量提交”流程：

```text
人工编辑草稿
  → 保存草稿
  → 标记为可提交
  → 批量提交可提交图片
  → 提交前统计确认
  → 旧正式标签备份
  → 草稿写入 labels/
  → review.json 标记 reviewed
  → commit_log.jsonl 记录日志
```

按钮含义：

- **保存草稿**：保存到 `.d1_annotations/drafts/`，正式 `labels/` 不变；
- **标记为可提交**：保存草稿并把状态置为 `draft_ready`，正式 `labels/` 不变；
- **需复核**：保存草稿并标记为 `needs_review`，不会进入“可提交”批量范围；
- **批量提交可提交图片**：先统计 `draft_ready` 图片数量和已有正式标签数量，确认后才批量写入正式 `labels/`。

批量提交时，已有正式标签会先备份到 `.d1_annotations/history/`，每张成功提交记录追加到：

```text
.d1_annotations/commit_log.jsonl
```

批量模型标注和批量提交是两个完全不同的操作：模型标注只写候选，批量提交只处理人工明确标记为 `draft_ready` 的草稿。

## 16. 多数据集统计与构建（第一版）

已加入 `datasets.py` 数据层和 Web 面板，输入数据集默认满足：标准 YOLO `images/{train,val,test}` + `labels/{train,val,test}`，类别 ID 已经是全局 ID。`detect_w1_fire` 已验证可直接扫描；其中存在部分 YOLO Segmentation 多边形标签，统计/构建时会转为外接 Detection 框，原始文件不修改。

### registry

复制示例并按需修改：

```bash
cp config/datasets_registry.example.yaml /path/to/datasets_registry.yaml
```

启动时传入：

```bash
d1-annotate \\
  --dataset /path/to/current_review_dataset \\
  --registry /path/to/datasets_registry.yaml
```

也可以在 Web 右侧“多数据集统计/构建”中登记数据集。登记只保存路径，不复制或修改源数据。

### 已支持接口

- `GET /api/datasets`：扫描登记数据集并返回各自统计；
- `POST /api/datasets`：登记一个已标准化数据集；
- `GET /api/datasets/summary`：汇总所有登记数据集的全局类别图片数和框数；
- `POST /api/datasets/build`：按选定全局类别和每类目标图片数构建新 YOLO 数据集。

构建输出包括：

```text
output/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
├── manifest.jsonl
└── statistics.json
```

构建会保留全局类别 ID，不做外部类别映射；一张多类别图片只复制一次，标签只保留选定类别。当前第一版使用多标签贪心抽样和固定随机种子，后续再增加来源比例、采集组切分、近重复检测和扫描缓存。



• 以后直接复制下面这段提示词，再补充本次修改内容即可。它明确了目标、上下文、边界和验收条件，符合 OpenAI Docs 的提示词建议。   
                                                                                                                              
  请将自动化标注项目从源服务器增量迁移到目标服务器，并完成验证。                                                              
                                                                                                                              
  源服务器：                                                                                                                  
  - SSH：sribd@10.20.14.86                                                                                                    
  - 源码目录：/home/sribd/D1_EDU/patrol_perception/annotation_tool                                                            
  - Portal 环境：d1-autolabel                                                                                                 
  - Portal 启动脚本：./run_portal.sh                                                                                          
  - SAM3 环境：d1-sam3                                                                                                        
  - SAM3 启动脚本：./sam3_service/run_thor.sh                                                                                 
                                                                                                                              
  目标服务器：                                                                                                                
  - SSH：sribd@10.20.14.24                                                                                                    
  - 部署根目录：/projects/sribd-apps/d1-autolabel                                                                             
  - Portal 容器：d1-autolabel-portal                                                                                          
  - SAM3 容器：d1-sam3                                                                                                        
  - Portal 地址：0.0.0.0:8765                                                                                                 
  - SAM3 地址：127.0.0.1:8783                                                                                                 
  - 目标硬件：x86_64 + NVIDIA Tesla T4                                                                                        
                                                                                                                              
  本次修改：                                                                                                                  
  【在这里说明修改了哪些功能、文件或提交；如果不清楚，就让你自行比较】                                                        
                                                                                                                              
  任务要求：                                                                                                                  
  1. 先只读检查两台服务器，对比 Git 提交、未提交修改、依赖文件和当前容器版本。                                                
  2. 判断本次修改影响 Portal、SAM3，还是两者都影响，只迁移必要组件。                                                          
  3. 为源码和部署前状态生成时间戳发布目录、文件清单和 SHA-256 校验值。                                                        
  4. 源机是 Jetson/Thor 环境，不要直接复制 Conda 环境或 ARM 镜像；在目标机重新构建兼容 x86_64、Tesla T4 和目标 CUDA 驱动的镜  
  像。                                                                                                                        
  5. 保留目标机现有数据、注册表、模型、缓存、配置和密钥文件，不覆盖或显示密钥内容。                                           
  6. 不要直接停止正式容器。先构建并启动 Canary 容器，完成检查后再切换。                                                       
  7. 保留旧容器和旧镜像作为回滚版本，切换失败时自动恢复。                                                                     
  8. 验证：                                                                                                                   
     - Portal 健康检查通过                                                                                                    
     - 数据集列表和页面正常                                                                                                   
     - SAM3 /health 正常且模型已加载                                                                                          
     - Portal 能调用 SAM3 完成一次测试                                                                                        
     - GPU、CPU、内存和日志无异常                                                                                             
     - 正式数据及标签文件哈希没有非预期变化                                                                                   
  9. 完成后报告：                                                                                                             
     - 源提交和工作区修改                                                                                                     
     - 实际迁移的文件                                                                                                         
     - 新旧镜像 ID                                                                                                            
     - 容器启动参数和挂载                                                                                                     
     - 测试结果                                                                                                               
     - 回滚命令                                                                                                               
     - 尚未解决的风险                                                                                                         
                                                                                                                              
  边界：                                                                                                                      
  - 不删除旧镜像、旧容器和历史发布记录。                                                                                      
  - 不修改正式数据，除非我明确授权。                                                                                          
  - 不在输出中显示密码、API Key 或环境变量值。                                                                                
  - 遇到无法确认的架构、CUDA、数据兼容性问题时，先停止切换并报告。                                                            
  - 在安全且属于上述范围的步骤中持续执行，不必逐步询问我；涉及删除数据或无法回滚的操作时必须询问。                            
                                                                                                                              
  如果只是一次小更新，可以使用简化版：                                                                                        
                                                                                                                              
  请按照既有迁移流程，将 10.20.14.86 上                                                                                       
  /home/sribd/D1_EDU/patrol_perception/annotation_tool                                                                        
  的最新修改增量迁移到 10.20.14.24。                                                                                          
                                                                                                                              
  先比较源码和当前部署，判断只需更新 d1-autolabel-portal，还是同时更新 d1-sam3。                                              
  保持现有数据、配置、密钥和模型挂载不变。先 Canary 验证，再切换正式容器，保留可回滚版本。                                    
  完成健康检查、接口测试、SAM3 调用测试和日志检查后，向我报告版本、改动、验证结果及回滚方法。                                 
  本次修改内容：【填写】                                                                                                      
                                                                                                                              
  尤其建议每次填写“本次修改内容”或 Git 提交号，这能显著减少误迁移。参考：OpenAI Docs：Prompting。                             
                                                                                                                              
─ Worked for 2m 38s ──────────────────────────────────────────────────────────────────────────────────────────────────────────                                                                                                                              
─ Conversation recap ─────────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                              
  目标是为反复修改、迁移10.20.14.86上的自动化标注项目准备提示词模板。已开始拟定，下一步需补全修改内容、迁移目标和验收条件。 
