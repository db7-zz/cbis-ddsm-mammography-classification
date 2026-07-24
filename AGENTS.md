# CBIS-DDSM 项目接手规则

本文件适用于本仓库中的所有Codex任务。开始工作前必须先完整阅读：

1. `docs/HANDOFF.md`
2. `docs/handoff_state.yaml`
3. 与当前阶段对应的spec和报告

## 当前阶段边界

- 阶段一至阶段四已经完成。
- 正式基线代码边界是Git标签`v1.0-baseline`；历史标签`phase-4-final-evaluation`指向同一提交。
- 阶段四官方测试推理已经执行一次，测试结果已经知晓。
- 阶段五spec已于2026-07-21获得用户明确批准，Gate A至Gate D均已完成。
- 阶段五最终候选为384输入、无随机增强；三随机种子确认仅使用train/validation。
- 用户已明确授权将`v1.0-baseline`标签和阶段五实验分支发布到GitHub；该授权不包含依赖安装、官方测试推理或阶段六架构改进。
- 阶段六spec已于2026-07-23获得用户明确批准，Gate 0至Gate D均已完成；全部正式运行只使用train/validation并记录`test_split_loaded=false`。
- 阶段六在EfficientNet-B0上冻结BiFPN C128/R1；CBAM的seed 42提升未通过三随机种子确认，validation经验最佳模型仍保留阶段五P5-E02。
- 用户已于2026-07-24授权将阶段六成果发布到最终分支`phase-6-final`，并删除其他阶段六相关分支；该授权不包含官方测试推理、依赖安装或阶段七EfficientNet-V2S实施。
- 阶段七尚未获批。开始任何V2S代码或训练前必须先起草阶段七spec并获得用户明确批准。

## 数据和实验纪律

- 当前基线固定为CBIS-DDSM calcification-only + whole mammogram。
- CC/MLO视图作为独立样本。
- `MALIGNANT=1`，`BENIGN`与`BENIGN_WITHOUT_CALLBACK=0`。
- train/validation/test必须保持patient_id两两无交集。
- 验证集用于模型选择和阈值；不得使用官方测试结果选择预处理、模型、epoch或超参数。
- 阶段五如获批准，只能加载train和validation；具体以批准后的`docs/phase5_preprocessing_resolution_spec.md`为准。
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

如果确需删除，只能一次删除一个明确路径的文件。不得删除、移动或暂存用户拥有的未跟踪目录`w1/`，除非用户明确说明。

## 当前未跟踪内容

- `w1/weekly report.docx`：用户文件，与模型代码无关；禁止擅自处理。
- `reports/CBIS-DDSM乳腺钼靶分类基线复现_现有成果报告.docx`：已生成的项目报告，是否纳入Git等待用户决定。

任何新对话都必须先运行`git status -sb`，不得假设工作区干净。
