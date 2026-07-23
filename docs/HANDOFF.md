# CBIS-DDSM项目对话迁移交接

生成日期：2026-07-22
用途：供全新Codex对话在不依赖旧聊天记录的情况下接手项目
状态：阶段一至阶段四冻结为v1.0-baseline；阶段五Gate A至Gate D完成并批准发布

## 1. 接手后的第一条原则

先确认本文件、`handoff_state.yaml`、Git状态和冻结产物一致，再按已批准的阶段五spec继续，不能扩展到阶段六。

阶段五已于2026-07-21获得实施授权，但未授权安装依赖、读取官方测试集、阶段六架构修改、commit或push。Gate C的P5-E01、P5-E02、P5-E03正式训练已于2026-07-22完成，全部只加载train/validation。

## 2. 仓库与Git真相

- 仓库：`https://github.com/db7-zz/cbis-ddsm-mammography-classification.git`
- 工作区：`D:\School Work\codex项目\surf`
- 默认分支：`main`
- 阶段四合并提交：`d6530e1f7a53fc131e3a15c6b39f7258599a49de`
- 阶段四实现提交：`56cdb1814e99523b797b7d7ce17ed5f058f95542`
- 正式基线标签：`v1.0-baseline`
- 历史阶段四标签：`phase-4-final-evaluation`（与正式基线指向同一提交）
- 阶段五实验分支：`codex/phase-5-experiments`
- 该分支通过本次交接提交发布到远端。在合并到`main`之前，从GitHub接手的新对话必须先检出该分支或查看对应PR，不能只读取旧的`main`。

已存在的阶段标签：

```text
phase-1-data-pipeline
phase-2-model-architecture
phase-3-training-baseline
phase-4-final-evaluation
```

## 3. 交接提交范围与本地保留文件

本次交接提交包含：

- `.gitignore`新增`backups/`；
- 新增阶段四冻结记录和冻结脚本；
- 新增中文研究计划V2；
- 新增中文阶段五spec；
- 新增本交接包；
- README新增交接入口。

以下两个本地目录不属于本次交接提交，仍需要特别保护：

- `reports/`：包含用户要求生成的现有成果Word报告；是否提交等待用户决定。
- `w1/`：包含用户的`weekly report.docx`，与本阶段代码无关。禁止删除、移动或默认暂存。

同一工作区接手后，`git status -sb`仍可能显示这两个未跟踪目录；全新clone则不会包含它们。接手后的第一条命令必须是：

```powershell
git status -sb
```

## 4. 项目真实目标与范围

严格基线为CBIS-DDSM钙化子集良恶性二分类：

- 输入：`256 × 256`；
- EfficientNet-B0；
- CNN feature map通过`1 × 1`卷积投影到`D=512`；
- 空间token + 二维正弦位置编码；
- Transformer Encoder：4层、8头、FFN=2048；
- mean pooling + LayerNorm；
- 输出一个logit；
- `BCEWithLogitsLoss(pos_weight=1.84)`；
- 验证集通过Youden's J选择阈值。

当前基线不包含BiFPN、CBAM、512输入、Grad-CAM++、多视图融合、分割或外部数据集。

## 5. 数据方案与本地路径

已批准并实际执行的数据方案：

- 仅使用官方calcification描述CSV；
- 使用`image file path`对应的whole mammogram；
- 不使用`cropped image file path`或ROI mask作为分类输入；
- CC/MLO每张视图作为独立样本；
- `MALIGNANT=1`；
- `BENIGN`和`BENIGN_WITHOUT_CALLBACK=0`；
- 官方test保留；官方train按`patient_id`划分train/validation；
- 三个split的患者交集均为0。

本地数据配置位于被Git忽略的`configs/data.local.yaml`。当前路径：

```text
数据根目录：D:/School Work/codex项目/数据集/dataset/cbis_ddsm
IDC metadata：D:/School Work/codex项目/数据集/dataset/metadata/metadata.csv
官方train CSV：D:/School Work/codex项目/数据集/dataset/official_csv/calc_case_description_train_set.csv
官方test CSV：D:/School Work/codex项目/数据集/dataset/official_csv/calc_case_description_test_set.csv
生成manifest：D:/School Work/codex项目/surf/artifacts/data
```

数据统计：

| Split | 图像 | 患者 | 阴性 | 阳性 |
|---|---:|---:|---:|---:|
| Train | 986 | 482 | 574 | 412 |
| Validation | 241 | 120 | 144 | 97 |
| Test | 284 | 151 | 165 | 119 |

全部1,511张DICOM已经成功解码。统一manifest SHA-256：

```text
f1c2a8709228087251bbd1b1d4a2ea55b2049c66458f3c8045096f460fb19758
```

“calcification-only + whole mammogram”是用户批准的数据方案，但由于原实验manifest与预测CSV缺失，必须继续标记为复现假设。

## 6. 阶段完成状态

### 阶段一：数据管线

已完成统一索引、patient-level split、Dataset/transforms、统计、全量解码验证和泄漏检查。权威文档：`docs/phase1_data_audit.md`。

### 阶段二：模型

已实现并验证EfficientNet-B0 + Transformer。参数量17,274,493。权威文档：`docs/phase2_model_spec.md`。

### 阶段三：正式训练

- 正式输出：`artifacts/training/baseline_seed42`
- 最佳epoch：12
- Validation ROC AUC：0.6586841352
- Validation PR AUC：0.5337626574
- Train loss：0.158105
- Validation loss：2.705716
- 训练约39.05分钟
- 早停正常触发
- `test_split_loaded=false`

训练表现出明显过拟合。曾发现min_delta错误影响数值最佳checkpoint保存；旧三轮运行已保留，修复后完整重训。本次正式结果来自修复后运行。

最佳checkpoint SHA-256：

```text
9b471183ea5d5b34365f2be0a5e5d401ef73dc56987847310679f364f1f3aabb
```

权威文档：`docs/phase3_training_spec.md`和`docs/phase3_training_report.md`。

### 阶段四：最终评估

- 正式输出：`artifacts/evaluation/baseline_seed42`
- 验证集冻结阈值：0.0482263415881329
- 官方测试图像：284
- 正式测试推理次数：1
- ROC AUC：0.7414311179
- PR AUC：0.6728686551
- Sensitivity：0.7058823529
- Specificity：0.6363636364
- Accuracy：0.6654929577
- F1：0.6387832700
- TN=105、FP=60、FN=35、TP=84

历史参考结果是ROC AUC 0.7869、PR AUC 0.6673、Sensitivity 80.6%。历史材料使用326条异常记录，本项目使用284张唯一全乳图像，两者样本口径不同。

权威文档：`docs/phase4_evaluation_spec.md`和`docs/phase4_evaluation_report.md`。

## 7. 正式产物冻结

阶段四代码边界已通过Git标签冻结。运行产物快照位于：

```text
backups/phase4_final_frozen_20260721
```

冻结结果：

- 35个备份文件；
- 398.57 MiB；
- 35/35个备份文件只读；
- 22/22个正式runtime源文件只读；
- 源/备份哈希错误为0。

冻结清单SHA-256：

```text
61176600394d4abdfc62aad075c9c2aa1567f9f03e654ff6117d89e6a867de30
```

权威记录：`docs/phase4_artifact_freeze_record.md`。可重复冻结脚本：`scripts/freeze_phase4_artifacts.py`。该脚本拒绝覆盖已存在快照，不要为了“测试”而解除只读或删除冻结目录。

快照与项目位于同一D盘，只能防止常规误覆盖；异盘灾备尚未完成。

## 8. 环境

- Conda环境：`surf`
- Python：3.11.15
- PyTorch：2.9.1+cu128
- torchvision：0.24.1+cu128
- CUDA：12.8
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，约12 GB显存
- Python路径：`C:\nonsystem\DevTools\conda\envs\surf\python.exe`

验证命令：

```powershell
& 'C:\nonsystem\DevTools\conda\envs\surf\python.exe' -m pytest -q
& 'C:\nonsystem\DevTools\conda\envs\surf\python.exe' scripts\verify_handoff.py
```

最近一次完整测试为`59 passed`。

## 9. 研究计划与下一阶段

原始研究计划保留在仓库外：

```text
C:\nonsystem\shoolwork\S3\surf\soph\材料\research-plan.md
```

它没有被修改，SHA-256为：

```text
9a25d44196a3b45dd3a0488e4dc2aec10a4efa3df07addb0e2f5b74737fb50e8
```

新的中文研究计划是`docs/research_plan_v2_post_baseline.md`。后续总体路线以V2为准；旧plan只作为历史设想和长期创新来源。

阶段五spec是`docs/phase5_preprocessing_resolution_spec.md`，当前状态为“已获用户批准，实施中”。其计划是先保持B0+Transformer架构固定，评估乳房区域crop+letterbox和256/384/512分辨率，再评估安全增强。阶段五明确禁止BiFPN、CBAM、V2S、Grad-CAM++和官方测试推理。

Gate A至Gate D已经完成。最终候选为384输入、无增强；seed 42、2026、3407的validation ROC AUC均值/样本标准差为0.684684/0.014927，PR AUC为0.581330/0.034546。全部运行均记录`test_split_loaded=false`。权威进度见`docs/phase5_progress_report.md`。阶段五通过`codex/phase-5-experiments`发布；下一步必须先审批阶段六spec，不得自动读取test或进入阶段六。

## 10. 已知问题与不可误读事项

1. 旧源码是B3、224输入、2-logit CE、无Transformer，不是本项目基线。
2. 历史N=326是异常记录数；当前N=284是唯一whole-mammogram图像数。
3. 经验训练负/正比约1.393，不是1.84；`pos_weight=1.84`来自锁定参考规格。
4. 当前概率阈值很低，表示校准偏移，不代表ROC/PR排序指标无效。
5. 当前模型过拟合明显，不能通过查看测试集选择改进方案。
6. 原始DICOM和本地配置不在Git中；新克隆需要重新配置`data.local.yaml`。
7. 备份在同一物理盘，不是完整灾难恢复方案。
8. 阶段四正式文件为只读；新实验必须使用新目录。

## 11. 新对话接手检查表

按顺序执行：

1. 阅读根目录`AGENTS.md`。
2. 阅读本文件和`docs/handoff_state.yaml`。
3. 运行`git status -sb`，确认没有把`w1/`当作项目文件。
4. 运行`git show phase-4-final-evaluation --no-patch`。
5. 运行`python scripts/verify_handoff.py`。
6. 运行完整pytest，当前预期59项通过。
7. 不运行训练或评估命令。
8. 向用户汇报已接手及阶段五Gate A/B进度。
9. 继续Gate C前确认用户当天允许的训练时间窗口；不得把阶段五批准扩展到阶段六。
10. 未获得commit/push批准时，不操作Git远端。

## 12. 交接完成的定义

一个新对话能够仅通过本仓库回答以下问题，即视为接手成功：

- 当前完成到哪个阶段？
- 正式基线数据、模型、训练和测试指标是什么？
- 哪些结果是历史参考，哪些是本次真实结果？
- 正式产物在哪里，如何验证未被修改？
- 当前工作区有哪些未提交或用户拥有的文件？
- 阶段五是否获批、允许改什么、禁止做什么？
- 下一步必须向用户请求什么决定？
