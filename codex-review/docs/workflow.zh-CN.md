# 审校工作流

Codex Review 基于冻结输入快照运行。运行目录保存 manifest、检查点、worker 输入与响应、判断、抽检材料和发布产物。续跑时使用同一运行目录；初始化后不要替换冻结输入。

## 模式

### 全书审校

1. **Scan：**逐段检查原文与译文的实质忠实度。
2. **Verify：**独立检查每一段，包括 scan 判为 clear 的段落。
3. **Fix：**针对已确认问题做最小修订，并保留原本正确的内容。
4. **独立盲审：**对每一段修订译文，以及未修改段落的固定 10% 抽样进行独立审查，不提供上游判断。若抽样发现问题，则扩大审查至其余段落。仍未解决的个案进入下述有次数上限的裁决流程。
5. **发布：**确认必要判断全部完成后，写入独立的审校状态和 EPUB。

### Attention 审校

1. **Triage：**使用初始上下文独立分类选中的个案。
2. **Expand：**为需要更多证据的个案收集更广的中性上下文并重新判断。
3. **Fix：**修复已确认问题。工作流允许首次修复以及最多两次追加修复。
4. **Accept：**独立审查修订后的文本。
5. **裁定未决问题：**裁定阶段可要求在全文索引中搜索、保留当前文本、选择拟议替换，或返回 `needs_user`。检索结果需要审阅，不能自动代替判断。
6. **独立接收审查：**裁定阶段选出的修改仍须经过单独的接收审查。
7. **发布：**只要还有 `needs_user`、pending 或 blocked 判断，就不能发布。使用 `decide` 记录用户选择保留或替换的决定及理由，然后续跑。

阶段依赖固定。模型分配在初始化时配置，之后被冻结；续跑时可以调整并发上限。各角色 worker 上限独立生效，且不能超过全局并发上限。示例将全局并发设为 6，scan、repair、judge 分别最多为 2、2、4。

对于已经完成的一次性审校，可将其最新已验收 CLI 状态冻结到新运行目录。旧原型的检查点不属于此工具的检查点格式，应保留其私有目录，不要用 `init` 覆盖。

## 命令

从仓库根目录运行命令。示例路径与 ID 应替换为合成值或本地值；不要把私有项目路径或原文加入提交示例。

```bash
python codex-review/review.py freeze --run-dir codex-review/runs/example --state ./private-project/state --source ./private-project/source.epub
python codex-review/review.py init --run-dir codex-review/runs/example --mode full --config codex-review/config.example.json
python codex-review/review.py run --run-dir codex-review/runs/example --config codex-review/config.example.json
python codex-review/review.py status --run-dir codex-review/runs/example
python codex-review/review.py results --run-dir codex-review/runs/example
python codex-review/review.py publish --run-dir codex-review/runs/example --output ./private-project/reviewed.epub
```

Attention 运行时使用 `--scope-json PATH` 冻结选定范围，再通过 `--mode attention` 初始化。可通过 `--terms-json PATH` 和 `--prior-run PATH` 提供术语和前次运行上下文。仅对等待用户输入的个案使用 `decide`，并提供保留决定或替换文件，以及理由。检查失败原因后，使用 `resume` 将阻塞任务重新加入队列，再使用 `run --resume` 重启 worker。单独的 `audit` 命令只能在所有审校判断解决后运行；它会生成分层、带随机种子的盲审 A/B 抽样文件，并单独保存私有映射，不会改变审校判断。两个文件都应保密。

```bash
python codex-review/review.py decide --run-dir codex-review/runs/example --id synthetic-segment --retain --reason "Reviewed against the source."
python codex-review/review.py resume --run-dir codex-review/runs/example --task-id synthetic-task
python codex-review/review.py run --run-dir codex-review/runs/example --config codex-review/config.example.json --resume
python codex-review/review.py audit --run-dir codex-review/runs/example --ratio 0.1 --seed 0
```

## 配置与登录

JSON 配置包含 `models` 和 `runtime` 两部分。`models` 将工作流角色（`scan`、`verify`、`triage`、`expanded`、`fix`、`accept`、`adjudicate`）映射到 Codex 模型 ID。`runtime` 设置 Codex 可执行文件、全局 worker 上限、各角色 worker 上限、任务超时和轮询间隔。见[`config.example.json`](../config.example.json)。

运行前先用 ChatGPT 账号登录 Codex CLI。运行器会检查登录方式并拒绝 API key 认证，再从 worker 进程环境中移除常见 API key 和 endpoint 环境变量。不要把凭据写入配置、命令参数、冻结数据或仓库。

## 运行数据与发布

建议将运行数据放在 `codex-review/runs/`，该目录已加入 Git 忽略规则。CLI 要求显式提供 `--run-dir`；自定义运行目录可能位于仓库外部，不受本项目 `.gitignore` 保护。运行目录可能包含原文摘录、译文和私有审校反馈。不要提交运行数据、审计映射或导出产物。

发布会在单独指定的输出路径创建新的审校状态和 EPUB，不会修改源 EPUB 或源项目状态。只要仍有 pending、blocked 或 `needs_user` 个案，发布就会阻止。运行目录仍保留审校证据和判断，发布后也应继续保密。

## 故障恢复

工作流记录任务和检查点，可从同一运行目录恢复中断任务。使用 `resume` 前先检查 `status` 和任务结果；恢复任务不得静默丢弃已完成工作或改变冻结原文。校验错误按配置策略重试。任务仍被阻塞时，保留诊断输出，先解决输入或任务问题再继续。
