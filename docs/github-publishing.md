# GitHub 发布与维护

公开仓库：<https://github.com/zh-zehui/patrol-perception>。

```bash
git clone https://github.com/zh-zehui/patrol-perception.git
cd patrol-perception
```

仓库发布代码、静态图示、合成演示截图与评测摘要。原始数据、模型、私有配置和密钥留在本地。

## 已配置的管理规则

| 设置 | 当前状态 |
| --- | --- |
| 默认分支 | `main` |
| 自动测试 | PR / main 推送运行 Python 3.10 和 3.12 CPU 测试 |
| 公开内容检查 | 检查受跟踪文件与完整可达 Git 历史中的常见密钥格式 |
| 分支保护 | 要求 PR、两个 CPU CI 检查通过、分支保持最新、讨论已解决 |
| 强推与删除 | `main` 禁止强推和删除，规则同样约束管理员 |
| 合并方式 | Squash merge；合并后自动删除功能分支 |
| 人工审批 | 当前为单维护者，不强制另一名审核者批准；`CODEOWNERS` 指向 `zh-zehui` |
| 依赖管理 | Dependabot 月度更新 GitHub Actions 与 Python 依赖 |
| 密钥保护 | GitHub Secret scanning 与 push protection 已启用 |
| 协作模板 | 问题、功能建议与 PR 模板已提交 |

首轮 GitHub CI 的两个 Python 版本各通过 154 项测试。持续状态以仓库 Actions 页面和首页徽章为准，不能把历史通过记录当成未来提交的测试结果。

## 日常更新

在工程目录创建短分支，修改、验证并提交：

```bash
git switch main
git pull --ff-only
git switch -c feat/my-change
# 修改代码或文档后：
python -m pytest
python scripts/check_publication.py --history
git add path/to/changed-file
git commit -m "feat: describe the concrete change"
git push -u origin feat/my-change
gh pr create --fill
gh pr checks --watch
gh pr merge --squash --delete-branch
```

不要直接向 `main` 推送修改。合并前检查 PR 内容和结果；模板与测试不能替代代码检查。CI 无需配置 API 密钥。

## 展示与版本

About 已设置项目简介与视觉感知相关 Topics。`v0.1.0` 定位为工程展示版，变更与限制见 [CHANGELOG.md](../CHANGELOG.md)。后续发布应说明验证环境、模型和数据版本、指标来源及未实现模块。

更新架构和状态时同步维护 `docs/status.md` 与 `scripts/generate_showcase.py`。更新指标时保存 JSON、来源元数据及限制说明，再重建图表；截图使用合成数据。完整操作见 [quickstart.md](quickstart.md)。

仓库尚未指定独立开源许可证；许可证需由权利人确定后补充。

配置依据：[GitHub Python CI 文档](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)、[分支保护 API](https://docs.github.com/en/rest/branches/branch-protection)、[Dependabot 配置参考](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)。
