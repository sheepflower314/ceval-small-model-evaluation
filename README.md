# C-Eval 小语言模型组内评测

四个方向共用固定的 C-Eval 本组自选子集、Qwen2.5-0.5B-Instruct 和评分协议。共用入口是 `evaluate_ceval.py`，正式对照显式使用严格输出基线 `--variant strict-prompt`。

## Fork 后快速安装环境

把自己的 Fork 克隆到 Windows 电脑，用 AI 助手打开项目根目录，然后复制整句：

> 在项目根目录按 `environment/README.md` 直接执行安装脚本，先检查本机的 64 位 Python 3.11 和 PyTorch 并优先复用可用环境，必要时在项目内新建环境并下载、校验模型，没有可用 CUDA 时用 CPU 模式，安装后运行 `.\environment\run.ps1` 验证中文推理，有 CUDA 时再运行 `.\environment\run.ps1 evaluate_ceval.py --variant strict-prompt --smoke` 完成 C-Eval 5 题验收，最后报告安装位置、实际设备和验收结果。

首次安装需要联网。手动操作或排查问题请看 [运行环境安装说明](environment/README.md)。

## 实验方向

| 方向 | 实验代码与配置 | 结果分析与复现记录 | 分支示例 |
|---|---|---|---|
| A 提示词设计 | `experiments/A_prompt/` | `analysis/A_prompt/` | `exp/a/prompt-wording` |
| B 少样本示例 | `experiments/B_few_shot/` | `analysis/B_few_shot/` | `exp/b/few-shot` |
| C 自检与修正 | `experiments/C_self_check/` | `analysis/C_self_check/` | `exp/c/self-check` |
| D 多次生成与答案选择 | `experiments/D_multi_generation/` | `analysis/D_multi_generation/` | `exp/d/vote` |

## 开始提交

使用 AI 协作时，先让助手读取 [C-Eval 组内协作 Skill](skills/ceval-team-workflow/SKILL.md)，并告诉它你的方向。识别 `AGENTS.md` 的工具会通过仓库根目录找到该 Skill。其他工具可直接使用这句话：

> 先读取并遵守 `skills/ceval-team-workflow/SKILL.md`；我是方向 [A/B/C/D] 的组员，按该方向的边界完成以下任务：[任务]。

首次克隆后，在仓库目录执行下列命令。Git 钩子用 Python 3 运行。Windows 上若 Python 不在 PATH，再将最后一条中的路径替换为自己的解释器。

```powershell
git config --local core.hooksPath .githooks
git config --local user.name "你的姓名"
git config --local user.email "你的 Git 托管邮箱"
git config --local ceval.python "D:/Anaconda3/envs/warehouse-gpu/python.exe"
```

每个任务从最新的 `main` 新建分支。以下以 A 为例，工作时明确暂存自己要提交的文件。

```powershell
git fetch origin
git switch -c exp/a/prompt-wording origin/main
# 完成修改后
git add -- experiments/A_prompt/ analysis/A_prompt/
git diff --cached --stat
git diff --cached
git commit -m "exp(a): compare prompt wording"
git push -u origin HEAD
```

推送后向 `main` 发起 PR。一项实验集中在一个 PR。公共推理、评分、依赖、固定题单和 Git 规则统一走独立的 `common/<task>` 分支。若实验依赖公共修改，先合公共 PR，再更新实验分支。既有写在共用入口里的实验变体，后续迁移也通过公共 PR 处理。

## 提交与合并规则

- `main` 只接收通过审核的 PR。提交钩子校验分支名和暂存内容，推送钩子拦截对已有远端 `main` 的直接更新。每位成员都应在自己的克隆里启用钩子。
- `exp/a/` 到 `exp/d/` 只修改上表对应的两个目录。修改其他方向目录、共用入口或固定题单会被拦截。分支名用于路径检查，成员之间的变更授权由 PR 审核确认。
- 固定题单由 `data/ceval/` 下的 `questions.jsonl`、`answers.jsonl` 和 `manifest.json` 组成。参考答案只供评分。公共评分或题单涉及更正时，PR 附影响范围、共用基线复跑结果和对既有结果的处理方式。正式对照使用同一设备、模型、题单和评分规则，并记录基线 commit、配置和运行命令。
- Git 只收录代码、配置、固定题单、复现说明和小型分析结果。模型、本地 Python 环境、原始下载数据、逐题原始运行日志和凭据保存在 Git 之外。运行原件留在 `runs/` 或团队共享存储，在分析记录中写明位置和实验编号。单个文件上限为 5 MiB。
- 自动检查覆盖路径、文件类型和大小、Python 语法及题单一致性。有 CUDA 的同学负责运行真实推理。公共变更至少跑 `--variant strict-prompt --smoke`，再由审核人查看原始输出和评分结果。

## 运行共享基线

新组员请先按 [运行环境安装说明](environment/README.md) 准备依赖和模型。安装后，在项目根目录用同一个入口验收和运行正式题单。

```powershell
.\environment\run.ps1 evaluate_ceval.py --variant strict-prompt --smoke
# 联调通过后跑正式题单
.\environment\run.ps1 evaluate_ceval.py --variant strict-prompt --owner "执行人"
```

固定题单的数据来源、版本、许可证和抽样规则见 `data/ceval/manifest.json`。

## 远端仓库设置

管理员将本地 `main` 推到团队选定的远端后，邀请四名成员。此仓库已提供 GitHub Actions 工作流和 PR 模板。使用 GitHub 时，在测试 PR 上先跑出 **Validate repository changes**，再将该检查设为 `main` 的必需状态检查。

在支持分支保护的仓库中，启用强制 PR、至少两名其他成员批准、推送新改动时清除旧批准、合并前分支与 `main` 同步、禁止强推和删除，并让管理员也遵守规则。仓库的规则、工作流与公共评测口径变更需要审核人逐项核对。使用其他托管平台时，需要在远端等价配置分支保护及 CI。
