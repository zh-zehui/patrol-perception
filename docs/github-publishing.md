# GitHub 发布与维护

公开仓库用于代码、静态图示、合成演示截图与评测摘要。原始数据、模型、私有配置和密钥留在本地。

## 首次上传

首先确认目标个人账号或组织，创建一个空仓库（不要自动初始化 README），然后在工程根目录执行：

```bash
python scripts/check_publication.py --history
python -m pytest
git remote add origin https://github.com/zh-zehui/patrol-perception.git
git push -u origin main
```

本项目目标账号为 `zh-zehui`，Fork 后请替换为自己的账号或组织。使用 GitHub CLI 的用户可在完成本机登录后创建并推送：

```bash
gh repo create zh-zehui/patrol-perception --public --source=. --remote=origin --push \
  --description "巡逻车视觉感知：数据集标注、YOLO 训练评测与离线事件输出"
```

二选一即可。当前文件提供操作入口，不代表远程仓库已经创建或推送成功。不要将访问令牌写入远程 URL、脚本或仓库文档。

## 仓库设置

- About 可填写上述简介，Topics 可用 `computer-vision`、`object-detection`、`yolo`、`annotation-tool`、`robotics`、`dataset`。
- 设置默认分支 `main`；在首次 CI 运行成功后为该分支配置合并规则，要求 PR 与 CPU 测试通过，禁用强推。以上是待在远端执行的设置。
- Issues 用模板收集复现问题，PR 描述填写行为变化和测试结果。
- CI 不需要 Secrets；不要为 CPU 测试添加模型服务密钥。
- 发布可展示版本时创建 `v0.x.y` 标签，说明已验证的环境与未实现模块；不要只为展示而宣称生产就绪。

## 内容更新

更新架构和状态时同步维护 `docs/status.md` 与 `scripts/generate_showcase.py`。更新指标时保存 JSON、来源元数据及限制说明，再重建图表；截图始终使用合成演示集，不能包含个人路径、服务密钥或实采敏感画面。

仓库尚未选择独立开源许可证。需要允许他人复用或再分发时，应先由权利人确定许可范围，再补充许可证文件。

配置依据：[GitHub Python CI 文档](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)、[Dependabot 配置参考](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)。
