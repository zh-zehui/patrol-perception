# D1 SAM 3 本地标注服务

该服务运行在独立的 Miniforge 环境 `d1-sam3` 中，复用 Jetson AGX Thor 上已经验证可用的 PyTorch 2.10 / CUDA 13.0 / SM 11.0 栈。Web 标注工具通过 `http://127.0.0.1:8783` 调用它，SAM 3 不会被安装进 `d1-autolabel` 环境。

## 权重下载与许可

当前机器使用 ModelScope 的公开镜像下载权重，不依赖 Hugging Face gated 权限。ModelScope 上可用的模型 ID 是 `facebook/sam3`：

```bash
source /home/sribd/miniforge3/etc/profile.d/conda.sh
conda activate d1-sam3
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/

python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    "facebook/sam3",
    local_dir="/home/sribd/models/sam3",
    allow_file_pattern=["sam3.pt", "config.json", "LICENSE", "README.md"],
)
PY
```

服务默认从 `/home/sribd/models/sam3/sam3.pt` 加载，也可以在启动前通过 `SAM3_CHECKPOINT` 指向其他本地权重。虽然下载不再需要 Hugging Face 审核，权重仍受仓库内 SAM License 约束。

当前服务使用 SAM 3 图像模型。SAM 3.1 的 Object Multiplex 改进主要面向多目标视频跟踪，不改变当前静态图片标注服务的默认选择。

## 环境

当前机器已经创建好环境。重新搭建时可从已验证的 Thor PyTorch 环境克隆：

```bash
source /home/sribd/miniforge3/etc/profile.d/conda.sh
conda create -n d1-sam3 --clone pytorch_cu130
conda activate d1-sam3
pip uninstall -y ultralytics ultralytics-thop opencv-python-headless scipy
pip install -r /home/sribd/D1_EDU/patrol_perception/annotation_tool/sam3_service/requirements.txt
```

SAM 3 官方当前仍导入 `pkg_resources`，因此这里固定 `setuptools<81`。`einops`、`pycocotools` 和 Triton 是当前代码实际会导入、但基础依赖声明不完整的组件。

## 启动服务

```bash
/home/sribd/D1_EDU/patrol_perception/annotation_tool/sam3_service/run_thor.sh
```

在另一个终端检查：

```bash
curl http://127.0.0.1:8783/health
curl -X POST http://127.0.0.1:8783/v1/load
```

`/v1/load` 会加载 `SAM3_CHECKPOINT` 指向的本地权重，不会访问 Hugging Face。服务只监听 `127.0.0.1`，不会暴露到局域网。

## 启动 Web 标注工具

```bash
conda activate d1-autolabel
source /home/sribd/.d1-autolabel-campus.env
d1-annotate \
  --dataset /path/to/dataset \
  --registry /home/sribd/D1_EDU/patrol_perception/annotation_tool/config/datasets_registry.yaml
```

Web 工具栏会显示：

```text
自动标注模型：视觉大模型 / SAM 3 本地模型
```

SAM 3 对每张图片只计算一次视觉特征，再按 `classes.yaml` 中每个类别的 `prompts` 执行文本概念检测。返回的像素坐标框经过同类 NMS 后转换为现有 YOLO 候选框格式，仍只写入 `.d1_annotations/candidates/`。
