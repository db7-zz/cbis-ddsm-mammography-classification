# 阶段五实验规格：乳房区域预处理与输入分辨率消融

状态：**已获用户批准，实施中**
创建日期：2026-07-21
批准日期：2026-07-21
是否已批准实施：**是**

## 1. 本规格的执行地位

本文件批准后，将成为阶段五的最高执行依据。

文件之间的优先级如下：

1. 用户在实施前给出的明确批准或修改意见；
2. 已批准的本阶段spec；
3. `research_plan_v2_post_baseline.md`中的总体研究路线；
4. 原始`research-plan.md`中的历史设想。

如果旧plan、新plan和本spec存在冲突，阶段五按照用户最终批准后的本spec执行。本文件已获批准，是当前阶段五实施依据；依赖安装、官方测试推理、阶段六内容、commit和push仍未授权。

## 2. 待批准的核心研究问题

阶段五只回答一个主要问题：

> 在保持EfficientNet-B0 + Transformer架构不变的情况下，确定性全乳区域预处理和更高输入分辨率能否提高验证集排序性能，并减轻当前基线的过拟合？

## 3. 阶段范围

### 3.1 本阶段允许开展

- 确定性去除非乳房黑色背景；
- 保留完整乳房的bounding box，并增加固定安全边距；
- 保持宽高比的缩放与padding；
- 比较`256`、`384`、`512`输入分辨率；
- 比较一组预先声明的钼靶安全增强策略；
- 进行CUDA显存烟雾测试、AMP和梯度累积适配；
- 仅使用验证集进行实验选择；
- 对最终候选方案进行多随机种子确认；
- 增加预处理、泄漏、张量形状和训练接口测试；
- 编写阶段五验证集实验报告。

### 3.2 本阶段明确禁止

- BiFPN或其他多尺度融合结构；
- CBAM或其他新注意力模块；
- EfficientNet-V2S或更换backbone；
- 调整Transformer层数、头数或隐藏维度；
- 将病灶ROI crop作为分类输入；
- Grad-CAM++、分割或定位指标；
- CC/MLO多视图融合；
- INbreast、VinDr-Mammo等外部数据集实验；
- 官方测试集推理或测试指标比较。

上述内容分别保留给阶段六至阶段八，避免一次加入多个变量导致无法归因。

## 4. 冻结的基线控制项

阶段五以下内容不得改变：

- CBIS-DDSM calcification-only数据方案；
- whole mammogram样本口径；
- 现有train/validation/test患者分配和manifest内容；
- 病理标签映射规则；
- ImageNet预训练EfficientNet-B0；
- 通道投影维度`512`；
- 四层Transformer Encoder和八个注意力头；
- FFN宽度`2048`、GELU和dropout `0.1`；
- token均值聚合和单logit输出；
- `BCEWithLogitsLoss(pos_weight=1.84)`；
- AdamW、初始学习率`1e-4`、weight decay `1e-4`；
- ReduceLROnPlateau和基于验证ROC AUC的早停；
- 最大50个epoch、梯度裁剪`1.0`和AMP；
- checkpoint选择规则：保存数值最高的验证ROC AUC；
- 如报告验证集分类指标，阈值仍使用Youden's J，并在完全并列时选择更高阈值。

阶段四tag和正式产物是只读对照。任何阶段五程序都不得写入：

```text
artifacts/training/baseline_seed42
artifacts/evaluation/baseline_seed42
```

## 5. 数据使用与测试集禁用

阶段五的数据集和训练入口只能加载`train.csv`与`val.csv`。

阶段五禁止读取：

- `test.csv`；
- 官方测试DICOM；
- 阶段四测试预测CSV；
- 阶段四测试指标JSON；
- 任何根据测试结果生成的候选模型排序信息。

每次正式运行必须检查：

- train/validation患者交集为空；
- train/validation manifest的SHA-256与阶段四冻结记录一致；
- 每个split内sample_id唯一；
- 两个类别均存在；
- validation transform完全确定性；
- 运行元数据明确记录`test_split_loaded=false`。

阶段五不产生任何官方测试指标。若后续确实需要评估，必须另写测试评估spec，并在阶段五方法完全冻结后单独获得批准。

## 6. 拟采用的预处理方案

### 6.1 强度处理

保留当前DICOM modality/VOI LUT处理、MONOCHROME反相规则、单图百分位截断和三通道转换。阶段五不调整强度截断百分位，避免与空间预处理变量混杂。

### 6.2 乳房区域mask

预处理必须是确定性的，不得使用病理标签或ROI mask决定裁剪边界：

1. 从LUT处理和强度截断后的灰度图生成前景mask。
2. 保留最大的八连通乳房区域。
3. 计算该区域的bounding box。
4. 在四个方向增加原图尺寸`2%`的固定安全边距。
5. 将扩展后的边界限制在原图范围内。
6. 如果前景比例或bounding box未通过有效性检查，则回退到完整原图，并记录具体原因。

拟定的有效性范围，等待审核：

- 前景面积占原图`15%`至`95%`；
- 裁剪后的乳房宽度和高度都不少于原始对应维度的`25%`；
- 安全边距裁剪不得丢弃任何已检测乳房前景像素；
- 输入为空、mask为空或几何异常时必须显式回退，禁止静默失败。

训练前必须进行确定性、标签盲的可视化审计。审计样本应覆盖：

- train和validation；
- CC和MLO；
- 左乳和右乳；
- 良性和恶性；
- 原始图像尺寸的最小值、最大值和中位范围；
- 正常裁剪和触发fallback的案例。

标签仅用于保证审计覆盖，不允许影响crop位置或算法参数。

### 6.3 等比例缩放与padding

- 保留裁剪后完整乳房的宽高比；
- 将较长边缩放到配置的输入尺寸；
- 较短边使用零值背景进行居中padding；
- 不再直接把乳房拉伸成正方形；
- 不允许利用病灶坐标裁剪局部区域。

第一轮阶段五实验不做左右乳方向统一。若后续需要laterality normalization，必须作为独立消融重新提出，不能混入P5-E01。

## 7. 数据增强方案

validation不使用任何随机增强。

第一轮分辨率对比也不使用增强，以便单独识别预处理和分辨率的贡献。

最佳分辨率冻结后，允许进行一个安全增强实验，拟定范围为：

- 旋转角度：`[-5°, +5°]`均匀采样；
- 水平和垂直平移：每个方向最多`3%`；
- 等比例缩放：`[0.95, 1.05]`；
- gamma：`[0.90, 1.10]`；
- 对比度因子：`[0.90, 1.10]`。

拟定概率，等待审核：

- 组合仿射变换概率：`0.5`；
- gamma概率：`0.3`；
- 对比度调整概率：`0.3`。

阶段五禁止：

- 随机病灶或patch crop；
- 弹性形变；
- MixUp和CutMix；
- 垂直翻转；
- 根据官方测试结果选择增强方式或增强强度。

## 8. 实验矩阵

| 实验ID | 乳房crop+letterbox | 输入尺寸 | 数据增强 | 实验目的 |
|---|---:|---:|---:|---|
| P5-E00 | 否 | 256 | 否 | 阶段四冻结对照，不重新训练 |
| P5-E01 | 是 | 256 | 否 | 单独评估空间预处理 |
| P5-E02 | 是 | 384 | 否 | 分辨率消融 |
| P5-E03 | 是 | 512 | 否 | 分辨率消融 |
| P5-E04 | 是 | E01-E03中最佳尺寸 | 是 | 评估安全增强贡献 |

P5-E00直接引用阶段四冻结基线，不重新训练，也不重新测试。

### 8.1 初筛规则

- 模型随机种子：`42`。
- 使用阶段一固定的患者划分。
- 允许完整训练和既定早停。
- 唯一主要选择指标：validation ROC AUC。
- validation PR AUC只作为次要指标，不得覆盖主要选择规则。

### 8.2 最终候选确认

初筛最佳配置使用三个模型随机种子：

```text
42、2026、3407
```

最终报告必须给出：

- 每个seed的验证ROC AUC和PR AUC；
- 三个seed的均值与标准差；
- best epoch、validation loss和过拟合趋势；
- 失败或OOM运行的保留记录。

如果两个初筛配置的验证ROC AUC差异小于`0.005`，默认选择分辨率更低、流程更简单的配置；除非多seed结果明确反转排序。该规则必须在训练前批准。

## 9. Batch size与显存策略

目标有效batch size固定为`16`：

| 分辨率 | 初始物理batch | 梯度累积步数 | 有效batch |
|---:|---:|---:|---:|
| 256 | 16 | 1 | 16 |
| 384 | 8 | 2 | 16 |
| 512 | 4 | 4 | 16 |

正式训练前，在RTX 5070 Ti Laptop GPU上运行前向和反向显存烟雾测试。

若发生OOM：

- 可以降低物理batch；
- 必须同步增加梯度累积，使有效batch保持16；
- 必须记录调整原因、实际batch、累积步数和峰值CUDA显存；
- 不得同时修改其他训练超参数。

预计token网格：

- `256`：约`8 × 8 = 64`个token；
- `384`：约`12 × 12 = 144`个token；
- `512`：约`16 × 16 = 256`个token。

Transformer注意力开销随token数平方增长，因此512分辨率必须通过显存门后才能正式训练。

## 10. 指标与选择规则

### 10.1 主要指标

- validation ROC AUC。

### 10.2 次要指标

- validation PR AUC；
- validation BCE loss；
- validation选择阈值下的Sensitivity和Specificity；
- best epoch与训练/验证泛化差距；
- 总训练时间；
- 峰值CUDA显存。

Accuracy、Precision、F1和校准统计可以在validation上报告，但不能改变主要模型选择结果。

阶段五不设置“必须提升多少才算完成”。如果所有实验合规执行后得到零提升或负结果，该结论仍然有效，必须如实记录。

## 11. 可复现性与产物目录

每次运行使用唯一且禁止覆盖的目录：

```text
artifacts/experiments/phase5/P5-E01_seed42/
artifacts/experiments/phase5/P5-E02_seed42/
artifacts/experiments/phase5/P5-E03_seed42/
artifacts/experiments/phase5/P5-E04_seed42/
```

每次运行必须保存：

- 完整解析后的data/model/train/experiment配置；
- Git commit和dirty状态；
- train/validation manifest SHA-256；
- 随机种子和确定性配置；
- epoch指标和TensorBoard event；
- 数值最佳checkpoint和last checkpoint；
- 最佳validation逐样本预测；
- 预处理fallback数量与原因；
- 训练耗时和峰值CUDA显存；
- 明确包含`test_split_loaded=false`的outcome。

`experiment_registry.csv`为每次完成或失败的运行保存一条不可覆盖记录。OOM和异常实验必须登记失败原因，不得通过删除目录隐藏。

## 12. 批准后预计修改的文件

批准后预计进行以下新增或局部修改：

- `configs/phase5/*.yaml`：阶段五实验配置；
- `src/cbis_ddsm/preprocessing.py`：乳房mask、crop和letterbox；
- `src/cbis_ddsm/dataset.py`：调用批准后的预处理管线；
- `src/cbis_ddsm/model.py`：只接受批准的可配置分辨率；
- `src/cbis_ddsm/training.py`：梯度累积与显存元数据；
- `scripts/audit_phase5_preprocessing.py`：确定性预处理审计；
- `scripts/run_phase5_experiment.py`：仅train/validation实验入口；
- `tests/test_preprocessing.py`：几何、fallback和确定性测试；
- `tests/test_phase5_training.py`：累积和测试集禁用测试；
- `docs/phase5_report.md`：阶段五验证集报告。

实施必须分批、最小化修改，不授权无关重构。

## 13. 阶段验收门

### Gate A：预处理审计

- 全部相关单元测试通过；
- crop不丢弃任何已检测乳房前景；
- 所有fallback均有原因记录；
- 确定性审计图已人工检查；
- 未加载任何test样本。

### Gate B：显存烟雾测试

- 每个分辨率前向/反向loss有限；
- 有效batch保持16；
- 峰值CUDA显存已记录；
- 烟雾测试不产生正式checkpoint或指标声明。

### Gate C：初筛矩阵

- P5-E01、P5-E02、P5-E03完成，或保留明确失败记录；
- 严格按照validation ROC AUC选择分辨率；
- P5-E04只在分辨率选择冻结后运行；
- 全过程未读取test。

### Gate D：多随机种子确认

- 最终候选三个seed全部完成；
- 均值、标准差和单次结果完整报告；
- 产物和哈希验证通过；
- 阶段五报告明确记录`test_split_loaded=false`。

阶段五以“仅验证集消融报告+用户审核”结束，不以官方测试集评估结束。

## 14. 需要用户审核的条款

请逐项批准或提出修改：

1. 阶段五范围及明确排除项；
2. 乳房mask有效范围和`2%`安全边距；
3. 数据增强范围及概率；
4. P5-E01至P5-E04实验矩阵；
5. seed `42、2026、3407`；
6. ROC AUC差异小于`0.005`时优先简单方案；
7. 有效batch size 16及梯度累积计划；
8. 阶段五不运行官方测试集；
9. 阶段五完成后再决定是否进入BiFPN阶段。
