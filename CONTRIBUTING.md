# 协作约定

`main` 保存可运行、可展示的版本。功能开发使用 `feat/<topic>`，修复使用 `fix/<topic>`，文档更新使用 `docs/<topic>`；完成后发起 Pull Request，说明具体行为和验证结果。

## 本地开发

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
python scripts/check_publication.py
```

CPU 测试不需要 API 密钥、外部数据集或 GPU。GPU 推理、训练与 SAM 3 另用与设备匹配的环境验证，不把硬件依赖加入基础标注环境。

## 变更要求

- 类别 ID、YOLO 坐标、候选/草稿/正式标签分层都是数据契约，修改时说明兼容性。
- 不把训练数据、真实密钥、模型权重、运行日志和个人数据集注册表提交到 Git。
- UI 改动提供合成数据截图；评测更新提供完整指标、日期、划分和来源。
- 对标签写入、数据集划分、类别映射等行为修改运行相关现有测试；新增高影响行为时补充有意义的回归测试。
- 使用简短明确的提交信息，如 `docs: refresh system overview`、`fix: preserve draft before commit`。

## 发布

合并前保持 CI 通过。发布说明记录新增能力、已知限制、验证环境及指标来源；确定版本后使用 `v0.x.y` 标签。权重和数据通过单独渠道管理，并明确对应的代码版本及来源。

GitHub 分支保护、发布和权限设置见[发布说明](docs/github-publishing.md)。
