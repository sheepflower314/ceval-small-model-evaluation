B 同学少样本实验

完整操作说明
先阅读同目录的“方向B少样本示例课程设计说明.txt”。该说明按当前 README.md 和组内协作规则编写，正式对照采用 strict-prompt 共享基线，B 的新入口放在 experiments/B_few_shot/。
下文 few-shot 命令对应已有的原始措辞两示例实现，可用于观察输入和联调。数量与顺序研究需先完成完整说明中的 B 独立入口。

负责内容
准备并核对少样本示例，运行 few-shot 配置，整理与零样本基线的对照。
分析稿和选出的案例保存在本目录。逐题原始日志统一保存在 runs 下。

文件入口
prepare_few_shot_examples.py 从题单清单指定的版本下载 dev 数据。
data/ceval/few_shot_examples.jsonl 保存每科两道示例及来源。
每科按原始题号排序，选取前两道完整且与正式题题干无重复的示例。
完整性检查和排重采用预先固定的规则，示例按文件顺序插入。
evaluate_ceval.py --variant few-shot 使用原始基线任务措辞和两道同科示例。

运行方式
先按 environment/README.md 安装环境，再在项目根目录 PowerShell 执行以下命令。已提供选定的示例。需要核对抽样过程时，运行 .\environment\run.ps1 prepare_few_shot_examples.py，缺失源文件时需联网。

.\environment\run.ps1 evaluate_ceval.py --variant few-shot --smoke --owner B

试跑后检查新建的 runs/few-shot-smoke-时间 目录中的完整输入和原始输出。
正式实验使用以下命令。

.\environment\run.ps1 evaluate_ceval.py --variant few-shot --owner B
.\environment\run.ps1 evaluate_ceval.py --variant baseline --owner B

正式运行前提交相关代码和示例，配置中会记录 commit 和源代码校验值。
少样本配置和上述 baseline 使用相同的原始提示措辞。分析准确率、格式有效率、耗时、输出长度和显存。

本次接入验收
试跑目录为 runs/few-shot-smoke-20261123-192913-856626。
共 5 题，答对 3 题，格式有效率 100%，平均单题耗时 0.5657 秒。
该结果用于检查输入和输出流程，正式结论需完成固定 200 题对照。
