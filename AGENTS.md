# CBIS-DDSM 项目接手规则

本文件适用于本仓库中的所有Codex任务。开始工作前必须先完整阅读：

1. `docs/HANDOFF.md`
2. `docs/handoff_state.yaml`
3. `docs/branching_policy.md`
4. 与当前阶段对应的spec和报告

## 当前阶段边界

- 阶段一至阶段四已经完成。
- 正式基线代码边界是Git标签`v1.0-baseline`；历史标签`phase-4-final-evaluation`指向同一提交。
- 阶段四官方测试推理已经执行一次，测试结果已经知晓。
- 阶段五Gate A至Gate D均已完成，最终结果冻结在`phase-5-final`。
- 阶段五最终候选为384输入、无随机增强；三随机种子确认仅使用train/validation。
- 阶段六实施分支为`phase-6-experiments`，从`phase-5-final`创建。
- 阶段六当前只完成分支初始化，尚未完成阶段六spec审批、架构改进、训练或评估。
- 未经新的明确批准，不得安装依赖、运行官方测试推理或开始阶段六实现。

## 分支管理

- `main`固定为阶段一至阶段四基线，必须与`v1.0-baseline`保持一致。
- 阶段五及以后不得合并回`main`。
- 阶段N实施必须从`phase-(N-1)-final`创建`phase-N-experiments`。
- 阶段N经用户确认完成后，才可发布为`phase-N-final`。
- 下一阶段只能从上一阶段的`phase-N-final`创建。
- README只能陈述对应分支已经完成的真实工作，不得把继承指标写成本阶段结果。
- 分支创建、改名、删除、commit和push均须遵守`docs/branching_policy.md`。
- 删除分支必须逐个指定名称并获得用户明确批准，禁止批量清理。

## 数据和实验纪律

- 当前基线固定为CBIS-DDSM calcification-only + whole mammogram。
- CC/MLO视图作为独立样本。
- `MALIGNANT=1`，`BENIGN`与`BENIGN_WITHOUT_CALLBACK=0`。
- train/validation/test必须保持patient_id两两无交集。
- 验证集用于模型选择和阈值；不得使用官方测试结果选择预处理、模型、epoch或超参数。
- 阶段五实验只能加载train和validation；阶段六在新spec批准前同样不得加载官方test。
- 禁止伪造日志、checkpoint、预测、哈希或指标。
- 未知或无法从原材料恢复的内容必须标记为“复现假设”。

## 冻结产物

以下目录是阶段四正式产物，不得覆盖或解除只读后原地修改：

```text
artifacts/training/baseline_seed42
artifacts/evaluation/baseline_seed42
backups/phase4_final_frozen_20260721
```

恢复或比较文件时必须先验证：

```text
backups/phase4_final_frozen_20260721/freeze_manifest.sha256
```

新实验必须使用新的、禁止覆盖的输出目录。

## 修改与审批流程

每个阶段实施前必须先向用户说明：

- 做什么；
- 为什么做；
- 修改哪些文件；
- 如何验证；
- 哪些内容不会做。

随后等待用户明确批准。不得把一个阶段的批准扩展到下一阶段。

未经用户明确批准，不得commit或push。暂存时必须显式列出文件，禁止在存在无关未跟踪内容时直接使用`git add -A`。

## 文件安全

禁止批量或递归删除文件或目录。不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

如果确需删除，只能一次删除一个明确路径的文件。不得删除、移动或暂存用户拥有的未跟踪目录`w1/`或`w2/`，除非用户明确说明。

## 当前未跟踪内容

- `w1/weekly report.docx`：用户文件，与模型代码无关；禁止擅自处理。
- `w2/`：用户拥有的未跟踪目录；内容和用途未经授权，禁止擅自处理。
- `reports/CBIS-DDSM乳腺钼靶分类基线复现_现有成果报告.docx`：已生成的项目报告，是否纳入Git等待用户决定。

任何新对话都必须先运行`git status -sb`，不得假设工作区干净。
