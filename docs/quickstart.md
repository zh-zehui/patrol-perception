# 运行入口

以下命令均在仓库根目录执行。标注演示见 [README](../README.md)。

## 离线检测与事件输出

使用已经安装匹配设备的 PyTorch 和 Ultralytics 的独立环境，准备合法取得的 YOLO26 权重及图片目录：

```bash
PYTHONPATH=src python -m patrol_perception.pipeline.offline_run \
  --source /path/to/ordered_images \
  --weights /path/to/model.pt \
  --device cpu \
  --out data/exports/demo
```

GPU 环境可用 `--device cuda:0`。程序按文件名排序读取单层图片目录，保存 `frame_000000.json` 等逐帧结果及汇总 `events.json`。它不会生成检测视频，也不连接 ROS 或相机。

IoU 跟踪和夜间逗留规则需要连续、按时间排序的图片；不相关图片不适合验证这些规则。夜间逗留当前按帧数计时，输出时间戳是处理时间，不能替代视频原始时间或生产级驻留计时。

检测器按权重自带类别名作映射。COCO 权重与 W2 权重的类别不同，必须检查自己的类别定义，不能通过修改名称伪造缺失类别能力。

## 训练

准备好合并后的 YOLO 数据集和初始权重后，可直接在已验证的 GPU 环境运行：

```bash
yolo detect train \
  model=/path/to/model.pt data=/path/to/data.yaml \
  imgsz=640 epochs=50 batch=16 \
  project=data/runs/detect name=my_run
```

已有 `scripts/train_detect_w1_fire.sh` 是开发机器上的专用包装脚本，包含本机 CUDA 环境切换路径；首次移植优先使用上面的显式命令，或自行修改包装脚本。训练细节见 [scripts/README.md](../scripts/README.md)。

## 统一评测

```bash
PYTHONPATH=src python scripts/eval_detect_run.py \
  --weights /path/to/best.pt \
  --data /path/to/data.yaml \
  --wave w2_joint \
  --config config/eval.yaml \
  --no-baseline
```

需要基线比较时把 `--no-baseline` 换成 `--baseline /path/to/baseline.pt`。默认配置中的基线文件需要自己准备；评测目录写入 `data/eval_runs/` 并被 Git 忽略。`--max-images N` 可执行抽样检查，结果展示时须注明抽样规模，不能冒充完整验证集评测。

## 更新展示资产

```bash
python -m pip install -r requirements-showcase.txt
python scripts/generate_showcase.py
python -m playwright install chromium
python scripts/capture_showcase.py
```

前一脚本生成 SVG 图和历史评测图；后一脚本使用临时合成数据与隔离配置启动实际界面、截图后关闭服务。它不连接现有门户、不读取现有数据集，也不调用模型。
