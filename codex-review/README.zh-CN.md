# Codex Review

Codex Review 是供翻译项目使用的本地、可续跑审校工具。它冻结原文快照，通过 Codex CLI 执行各阶段审校，在运行目录记录判断，并且只在所有必要判断都已解决后发布。

## 环境要求

- Python 3.10 或更高版本。
- macOS 或 Linux；运行器使用 POSIX 文件锁和进程组。
- 已安装并已用 ChatGPT 账号登录的 Codex CLI。
- 本地 Wenyi 项目及其源 EPUB。

运行器会确认 Codex CLI 已用 ChatGPT 登录，并拒绝 API key 登录方式；没有 API key 或 endpoint 配置项。请将冻结输入和全部运行数据保存在私有本地目录中。

## 快速开始

先运行一次 `uv sync --locked`。在仓库根目录使用其虚拟环境执行示例，或将 `python` 替换为 `uv run --no-sync python`。`--state` 指向含 `manifest.json` 和 `chapters/` 的 CLI 目标语言目录，不是总 `state_dir`，也不是 Web/Desktop 工作区。

将占位输入替换为本地项目路径后，冻结运行数据，再初始化并运行：

```bash
python codex-review/review.py freeze --run-dir codex-review/runs/example --state ./private-project/state --source ./private-project/source.epub
python codex-review/review.py init --run-dir codex-review/runs/example --mode full --config codex-review/config.example.json
python codex-review/review.py run --run-dir codex-review/runs/example --config codex-review/config.example.json
python codex-review/review.py status --run-dir codex-review/runs/example
```

`full` 会审校全部段落。对已有审校中选出的段落使用 `attention` 模式进行分流。使用 `results` 查看运行结果；未解决的 attention 个案必须先作出决定，才能发布。通过 `publish` 导出到单独的 EPUB 路径。阶段、恢复、决定、盲审和发布流程见[工作流指南](docs/workflow.zh-CN.md)。

配置示例指定模型 ID 和 worker 并发上限。可以复制为私有的 `*.local.json` 文件再修改；该命名模式已加入 Git 忽略规则。不要在配置中加入真实 key 或 endpoint 地址。

## 数据与隐私

建议将运行目录放在 `codex-review/runs/`，该目录已加入 Git 忽略规则。运行器要求显式提供 `--run-dir`，并会为新建运行目录写入自身也被忽略的 `.gitignore`，包括自定义位置。运行目录包含原文摘录、拟议译文和修订译文、任务结果、检查点与发布产物。请将其视为私有项目数据。

发布会写入独立的审校状态和 EPUB 文件，不会覆盖源 EPUB 或源项目状态。审校步骤和输出契约见[工作流指南](docs/workflow.zh-CN.md)。
