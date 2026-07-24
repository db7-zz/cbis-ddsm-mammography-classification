# 阶段六实验规格：EfficientNet-B0上的BiFPN与CBAM模块消融

状态：**已获用户批准并完成Gate 0至Gate D；CBAM多seed均值未提升，validation经验最佳模型保留阶段五候选**
创建日期：2026-07-23
批准日期：2026-07-23
是否已批准实施：**是**

## 1. 本规格的执行地位

本文件根据`docs/research_plan_v2_post_baseline.md`重新生成，用于规定阶段六允许开展的架构消融。只有用户明确批准后，才允许创建阶段六代码、配置、测试或运行实验。

文件之间的优先级如下：

1. 用户在实施前给出的明确批准或修改意见；
2. 用户批准后的本阶段spec；
3. `docs/research_plan_v2_post_baseline.md`中的总体研究路线；
4. 阶段五已批准spec和阶段五冻结结果；
5. 原始`research-plan.md`中的历史设想。

本spec获批只授权其中明确列出的阶段六工作，不自动授权依赖安装、官方测试推理、EfficientNet-V2S迁移、阶段七、阶段八、commit或push。

## 2. 核心研究问题

阶段六回答以下三个相互关联但可分别归因的问题：

1. 在阶段五最终候选上，BiFPN多尺度特征融合能否提高validation排序性能？
2. 在不加入BiFPN时，CBAM能否独立提高validation排序性能？
3. 使用已经冻结的BiFPN与CBAM配置时，两者组合是否产生互补收益？

阶段六最终形成：

- BiFPN在EfficientNet-B0上的独立消融结论；
- CBAM在EfficientNet-B0上的独立消融结论；
- BiFPN与CBAM组合效果；
- 可供阶段七重新审批后迁移至EfficientNet-V2S的冻结模块配置。

## 3. 阶段范围

### 3.1 本阶段允许开展

- 继续使用阶段五最终候选的`384 × 384`输入、乳房crop+letterbox和无随机增强；
- 从EfficientNet-B0提取P2-P5特征；
- 将P2-P5投影到统一通道宽度；
- 实现可学习加权的双向BiFPN；
- 比较预先声明的BiFPN重复次数`R=1`和`R=2`；
- 在进入Transformer前使用固定token压缩策略；
- 在统一的`512 × 12 × 12`特征接口处加入一套预先声明的CBAM；
- 分别训练BiFPN、CBAM以及BiFPN+CBAM模型；
- 使用固定seed 42进行初筛；
- 对阶段六validation最佳候选按本spec规则进行多随机种子确认；
- 进行架构、参数量、梯度、显存和数据禁用审计；
- 编写仅基于train/validation的阶段六报告。

### 3.2 本阶段明确禁止

- EfficientNet-V2S或任何backbone切换；
- 根据阶段六结果继续搜索BiFPN通道宽度、卷积核或任意重复次数；
- 搜索CBAM插入位置、reduction ratio、空间卷积核或多个CBAM数量；
- 改变Transformer层数、头数、隐藏维度、FFN宽度或聚合方式；
- 改变阶段五预处理、输入分辨率或数据增强；
- Grad-CAM++、ROI定位、分割或多任务学习；
- CC/MLO多视图融合；
- 外部数据集；
- 官方test manifest、官方test DICOM、阶段四test预测或test指标读取；
- 依赖安装；
- 阶段七或阶段八实施；
- 未经单独批准的commit或push。

## 4. 冻结的阶段五控制项

阶段六以下内容不得改变：

- CBIS-DDSM calcification-only + whole mammogram数据方案；
- CC/MLO视图作为独立样本；
- patient-level train/validation划分；
- `MALIGNANT=1`，`BENIGN`和`BENIGN_WITHOUT_CALLBACK=0`；
- train 986张、validation 241张；
- 阶段五确定性乳房crop+letterbox；
- 输入尺寸`384 × 384`；
- train和validation均不使用阶段五安全随机增强；
- ImageNet预训练EfficientNet-B0并全量微调；
- Transformer维度`D=512`、4层、8头、FFN=2048；
- 二维正弦位置编码、mean pooling、LayerNorm和单logit输出；
- `BCEWithLogitsLoss(pos_weight=1.84)`；
- AdamW、初始学习率`1e-4`、weight decay `1e-4`；
- ReduceLROnPlateau、最大50 epoch、AMP和梯度裁剪`1.0`；
- 有效batch size固定为16；
- validation ROC AUC作为唯一主要选择指标。

阶段五冻结对照为：

```text
P5-E02_seed42
输入：384 × 384
预处理：乳房crop+letterbox
随机增强：无
Validation ROC AUC：0.6995632875
Validation PR AUC：0.6162027464
```

阶段五三个seed的validation ROC AUC均值和样本标准差为`0.6846840397 ± 0.0149272049`。阶段六不得重新训练P6-E00，也不得使用阶段四官方测试结果参与任何选择。

## 5. 数据使用与实施前置条件

阶段六训练入口只能解析并加载：

```text
train.csv SHA-256: 3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716
val.csv   SHA-256: 15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0
```

每次正式运行必须检查：

- train/validation患者交集为0；
- 每个split内sample_id唯一；
- train和validation均包含两个类别；
- validation transform完全确定性；
- validation预测恰好覆盖241个唯一sample_id；
- 运行元数据明确记录`test_split_loaded=false`；
- 配置和入口中不存在test路径参数。

当前worktree未包含Git忽略的`artifacts/data/manifest.csv`、`train.csv`和`val.csv`，完整pytest因此为`58 passed, 1 failed`。在实施阶段六前必须先以只读、可审计方式恢复或明确挂载锁定的train/validation产物，并重新达到预期完整测试`59 passed`。不得为了修复该环境问题而生成新split、复制test manifest或读取官方test。

## 6. EfficientNet-B0的P2-P5定义

在torchvision 0.24.1的EfficientNet-B0中，固定使用每个目标分辨率下语义最深的以下端点：

| 层级 | torchvision端点 | 相对步长 | 384输入下形状 |
|---|---|---:|---|
| P2 | `features[2]` | 4 | `24 × 96 × 96` |
| P3 | `features[3]` | 8 | `40 × 48 × 48` |
| P4 | `features[5]` | 16 | `112 × 24 × 24` |
| P5 | `features[8]` | 32 | `1280 × 12 × 12` |

端点不得在训练后根据validation结果调整。Gate A必须使用合成`1 × 3 × 384 × 384`张量重新记录每个端点的实际形状。

## 7. BiFPN具体设计

### 7.1 统一通道宽度

P2-P5分别通过独立的：

```text
1 × 1 Conv2d(bias=false) → BatchNorm2d → SiLU
```

投影到固定通道宽度`C=128`。本阶段不搜索`C=64`、`160`、`256`或其他宽度。

### 7.2 加权融合

每个融合节点使用非负快速归一化权重：

```text
normalized_weight_i = ReLU(weight_i) / (epsilon + Σ ReLU(weight_j))
epsilon = 1e-4
fused = Σ normalized_weight_i × input_i
```

所有融合权重初始化为`1.0`。融合后固定使用：

```text
3 × 3 depthwise Conv2d(bias=false, padding=1)
→ 1 × 1 pointwise Conv2d(bias=false)
→ BatchNorm2d
→ SiLU
```

### 7.3 双向路径

每次BiFPN重复严格包含：

自顶向下：

```text
T5 = I5
T4 = SepConv(Fuse(I4, Up(T5)))
T3 = SepConv(Fuse(I3, Up(T4)))
T2 = SepConv(Fuse(I2, Up(T3)))
```

自底向上：

```text
O2 = T2
O3 = SepConv(Fuse(I3, T3, Down(O2)))
O4 = SepConv(Fuse(I4, T4, Down(O3)))
O5 = SepConv(Fuse(I5, Down(O4)))
```

其中：

- `Up`固定为nearest-neighbor 2倍上采样；
- `Down`固定为`3 × 3 MaxPool2d(stride=2, padding=1)`；
- 下一次重复将上一重复的`O2-O5`作为新的`I2-I5`；
- 不增加没有输入贡献的单输入节点；
- 不使用可变形卷积、注意力卷积或额外残差分支。

### 7.4 重复次数

阶段六只比较：

- `R=1`；
- `R=2`。

使用seed 42和validation ROC AUC在两者之间冻结一个BiFPN配置。如果ROC AUC差异小于`0.005`，选择计算更简单的`R=1`。冻结后不得为组合实验重新选择重复次数。

## 8. Transformer token压缩策略

禁止将原始P2-P5全部空间位置直接拼接为token。未经压缩时共有：

```text
96² + 48² + 24² + 12² = 12,240 tokens
```

固定压缩流程为：

1. 将BiFPN输出`O2-O5`分别池化到`12 × 12`；由于四层尺寸均可被12整除，正式实现固定使用数学等价且支持CUDA确定性反向传播的`AvgPool2d`，kernel/stride依次为`8、4、2、1`，不使用CUDA反向非确定的`AdaptiveAvgPool2d`；
2. 按`O2、O3、O4、O5`顺序沿通道维拼接；
3. 得到`512 × 12 × 12`特征；
4. 使用`1 × 1 Conv2d(512→512, bias=false) → BatchNorm2d → SiLU`进行尺度混合；
5. 按行优先顺序展平为`144 × 512`token；
6. 加入与阶段五相同的二维正弦位置编码；
7. 输入冻结配置的4层Transformer。

因此BiFPN模型的Transformer token数固定为144，与阶段五384模型完全相同。阶段六不增加scale token、CLS token或独立的多尺度Transformer分支。

## 9. CBAM具体设计

### 9.1 唯一插入位置

CBAM只允许插入在进入Transformer之前的`512 × 12 × 12`特征上：

- P6-E02：EfficientNet-B0最终`1280 × 12 × 12`特征经阶段五`1 × 1 Conv 1280→512`投影后，再经过CBAM；
- P6-E03：BiFPN多尺度池化、拼接和`1 × 1 Conv 512→512`尺度混合后，再经过同一CBAM；
- CBAM输出随后展平为144个token。

不得把CBAM插入backbone内部、每个P层、每个BiFPN节点或Transformer层内。

### 9.2 通道注意力

固定使用：

- 全局平均池化和全局最大池化两条分支；正式实现使用显式`mean`和`amax`以保持CUDA确定性反向传播；
- 两条分支共享MLP；
- reduction ratio固定为`16`；
- 对512通道，隐藏通道固定为`32`；
- 共享MLP为`1 × 1 Conv 512→32 → ReLU → 1 × 1 Conv 32→512`；
- 两条分支相加后使用sigmoid；
- 得到的权重与输入逐通道相乘。

### 9.3 空间注意力

通道注意力之后固定使用：

- 沿通道维分别计算平均值和最大值；
- 拼接为2通道空间图；
- 使用`7 × 7 Conv2d(2→1, padding=3, bias=false)`；
- sigmoid后与特征逐位置相乘。

CBAM顺序固定为“通道注意力→空间注意力”。不增加额外残差、BatchNorm、dropout或第二个CBAM。

## 10. 实验矩阵

| 实验ID | Backbone | BiFPN | CBAM | Token数 | Seed | 目的 |
|---|---|---|---|---:|---:|---|
| P6-E00 | B0 | 否 | 否 | 144 | 42 | 引用P5-E02冻结对照，不重训 |
| P6-E01A | B0 | C128、R1 | 否 | 144 | 42 | BiFPN重复次数初筛 |
| P6-E01B | B0 | C128、R2 | 否 | 144 | 42 | BiFPN重复次数初筛 |
| P6-E02 | B0 | 否 | 是 | 144 | 42 | CBAM独立贡献 |
| P6-E03 | B0 | E01冻结配置 | 是 | 144 | 42 | BiFPN与CBAM组合效果 |

执行顺序必须为：

1. 引用P6-E00；
2. 完成P6-E01A和P6-E01B并冻结BiFPN重复次数；
3. 完成P6-E02；
4. 使用已经冻结的BiFPN与CBAM定义运行P6-E03；
5. 按第11节规则决定是否进行多随机种子确认。

P6-E03不得在看到P6-E02结果后改变CBAM位置或参数。

## 11. 指标与选择规则

### 11.1 主要指标

- validation ROC AUC。

### 11.2 次要指标

- validation PR AUC；
- validation BCE loss；
- best epoch；
- train/validation loss和ROC AUC差距；
- 训练时间；
- 峰值CUDA显存；
- 参数量和每样本推理时间。

PR AUC、loss、阈值分类指标和训练速度不得覆盖主要选择规则。

### 11.3 BiFPN内部选择

先比较P6-E01A与P6-E01B：

- ROC AUC差异达到或超过`0.005`时选择较高者；
- 差异小于`0.005`时选择`R=1`；
- 选择完成后冻结BiFPN配置，再运行P6-E03。

### 11.4 阶段六seed 42选择

在P6-E00、冻结后的P6-E01、P6-E02和P6-E03之间比较：

- ROC AUC差异达到或超过`0.005`时选择较高者；
- 差异小于`0.005`时优先选择总可训练参数更少者；
- 参数量相同时优先选择新增模块更少者；
- 仍相同时按`P6-E00 → P6-E02 → P6-E01 → P6-E03`顺序选择。

### 11.5 多随机种子确认

如果seed 42阶段六最佳非对照配置比P6-E00至少高`0.005`，则固定全部配置并补充：

```text
seed 2026
seed 3407
```

最终报告比较阶段六候选与阶段五对照在seed 42、2026、3407下的validation ROC AUC均值、样本标准差和同seed差值。

- 阶段六候选均值比阶段五均值至少高`0.005`时，阶段六候选成为validation经验最佳模型；
- 均值差异小于`0.005`时保留更简单的阶段五候选；
- 如果seed 42没有达到进入多seed确认的门槛，不追加阶段六训练，直接保留阶段五已有三seed结论。

无论结果是否提升，阶段六仍冻结：

- P6-E01A/P6-E01B中选出的BiFPN重复次数；
- 本spec规定的CBAM位置、reduction ratio和空间卷积核；
- 组合模型的完整结构定义。

这些冻结定义只作为阶段七spec的候选输入，不表示阶段七已经获批。

## 12. 训练与早停规则

所有阶段六正式训练继续使用：

```text
max_epochs: 50
optimizer: AdamW
learning_rate: 0.0001
weight_decay: 0.0001
loss: BCEWithLogitsLoss
pos_weight: 1.84
scheduler: ReduceLROnPlateau
scheduler_factor: 0.5
scheduler_patience: 3
min_learning_rate: 0.000001
early_stopping_monitor: val_roc_auc
early_stopping_mode: max
early_stopping_patience: 10
early_stopping_min_delta: 0.001
gradient_clip_norm: 1.0
mixed_precision: true
deterministic: true
```

checkpoint规则：

- `best.pt`始终保存数值最高的validation ROC AUC；
- `early_stopping_min_delta`只影响耐心计数，不得影响数值最佳checkpoint保存；
- 同一实验目录禁止覆盖或自动续写不匹配的配置；
- 恢复训练必须验证配置哈希、seed和checkpoint来源一致。

## 13. 参数量、显存和batch上限

### 13.1 参数量

- 阶段五B0+Transformer对照：17,274,493个可训练参数；
- 阶段六任一候选的总可训练参数硬上限：18,000,000；
- Gate A必须记录每个候选的总参数和可训练参数；
- 超过上限必须停止，不得通过临时减小Transformer或backbone来规避。

### 13.2 CUDA显存

目标GPU为约12 GB显存的RTX 5070 Ti Laptop GPU。

- 正式配置的`torch.cuda.max_memory_allocated`硬上限：8 GiB；
- 初始物理batch为8、梯度累积2；
- 若OOM或超过8 GiB，依次尝试`4 × 4`、`2 × 8`、`1 × 16`；
- 有效batch始终保持16；
- 降低物理batch时不得同时修改学习率、模型结构或其他训练超参数；
- 物理batch 1仍OOM或超过8 GiB时，该实验登记为Gate B失败，不进入正式训练。

显存烟雾测试只使用合成张量，不加载任何数据split，不下载预训练权重，不写正式checkpoint，也不声明validation指标。

## 14. 失败、OOM和异常记录

每次正式或失败运行必须使用唯一目录并写入实验登记表。失败记录至少包含：

- 实验ID、配置和配置SHA-256；
- Git HEAD与dirty状态；
- seed和确定性设置；
- 物理batch、梯度累积和有效batch；
- 失败阶段、最后完成epoch或batch；
- `failure_type=oom`、`nan`或`exception`；
- 异常摘要；
- 峰值CUDA显存；
- 已生成产物的哈希；
- `test_split_loaded=false`；
- 指标字段保持空值，不得填入伪造的0或最佳历史值。

OOM或负结果不得通过删除目录、修改registry历史行或覆盖输出隐藏。

## 15. 输出目录与产物

阶段六使用：

```text
artifacts/experiments/phase6/architecture_audit_v2.json
artifacts/experiments/phase6/memory_smoke_v1.json
artifacts/experiments/phase6/P6-E01A_BiFPN_R1_C128_seed42/
artifacts/experiments/phase6/P6-E01B_BiFPN_R2_C128_seed42/
artifacts/experiments/phase6/P6-E02_CBAM_seed42/
artifacts/experiments/phase6/P6-E03_BiFPN_CBAM_seed42/
artifacts/experiments/phase6/experiment_registry.csv
```

多seed目录沿用选中实验ID并替换seed。每个正式运行保存：

- 完整解析配置和配置哈希；
- Git元数据；
- train/validation manifest哈希；
- 架构审计结果；
- epoch级指标和TensorBoard event；
- `best.pt`和`last.pt`；
- 最佳validation逐样本预测；
- 参数量、训练时间和峰值显存；
- `outcome.json`；
- 所有关键产物SHA-256；
- `test_split_loaded=false`。

输出目录存在时默认拒绝运行，不允许自动清空或覆盖。

## 16. 批准后预计修改的文件

批准后预计新增：

```text
configs/phase6/manifest_lock.yaml
configs/phase6/P6-E01A_BiFPN_R1_C128_seed42.yaml
configs/phase6/P6-E01B_BiFPN_R2_C128_seed42.yaml
configs/phase6/P6-E02_CBAM_seed42.yaml
configs/phase6/P6-E03_BiFPN_CBAM_seed42.yaml
src/cbis_ddsm/phase6_model.py
src/cbis_ddsm/phase6.py
scripts/audit_phase6_architecture.py
scripts/run_phase6_experiment.py
tests/test_phase6_model.py
tests/test_phase6_training.py
docs/phase6_progress_report.md
```

多seed配置只在达到第11.5节门槛后新增。优先通过新的阶段六模块复用现有训练接口，不修改`src/cbis_ddsm/model.py`中的阶段五模型定义，不覆盖阶段四或阶段五正式产物。

如实施中发现必须修改上述范围外的共享文件，应停止并重新向用户说明原因、影响和验证方案。

## 17. 验收Gate

### Gate 0：实施就绪

- 用户明确批准本spec和阶段六实施范围；
- 阶段六从提交`4e3b807421933ae3f4cb70bcce65263f2757e53c`及已审核文档改动开始；
- 锁定train/validation manifest可只读访问且哈希一致；
- 完整pytest达到预期`59 passed`；
- 阶段四冻结边界未被修改；
- 未解析或加载官方test。

### Gate A：架构与单元测试

- P2-P5端点和形状与第6节一致；
- BiFPN每层输出形状、融合权重归一化和R1/R2重复正确；
- CBAM位置、通道注意力和空间注意力与第9节一致；
- 所有候选输出单logit且Transformer token数均为144；
- 参数量不超过18,000,000；
- 合成输入前向、反向和梯度均为有限值；
- ImageNet B0预训练权重能够严格、可审计地加载；
- 阶段五模型既有行为和测试不回归。

### Gate B：CUDA显存烟雾测试

- P6-E01A、P6-E01B、P6-E02和P6-E03均完成前向、反向和一次优化器更新；
- loss和梯度有限；
- 有效batch保持16；
- 峰值显存不超过8 GiB；
- 实际物理batch和累积步数冻结到正式配置；
- 不加载任何数据split，不写checkpoint，不声明实验指标。

### Gate C：seed 42初筛

- P6-E01A和P6-E01B完成或保留合规失败记录；
- 按预先声明规则冻结BiFPN重复次数；
- P6-E02和P6-E03完成或保留合规失败记录；
- 每次预测覆盖241个唯一validation sample_id；
- 从逐样本预测独立重算ROC AUC和PR AUC一致；
- 只按validation ROC AUC及预先声明的简化规则选择；
- 全部运行记录`test_split_loaded=false`。

### Gate D：多seed确认与报告

- 仅在达到第11.5节门槛时运行seed 2026和3407；
- 给出单seed、均值、样本标准差和同seed差值；
- 报告best epoch、loss、过拟合趋势、训练时间、参数量和显存；
- 失败、OOM和负结果完整保留；
- registry、checkpoint、预测和配置哈希核验通过；
- 明确区分阶段六validation经验最佳模型与阶段七规定核心方法；
- 阶段六报告经用户审核；
- 未读取或运行官方test。

## 18. 需要用户审核的条款

请逐项批准或提出修改：

1. 阶段六只在EfficientNet-B0上完成BiFPN、CBAM及组合消融；
2. P2-P5端点和384输入下的具体形状；
3. BiFPN统一通道`C=128`及`R=1/R=2`初筛；
4. BiFPN加权融合、上下采样和深度可分离卷积定义；
5. 四尺度池化到`12 × 12`、拼接并保持144个Transformer token；
6. CBAM唯一插入在`512 × 12 × 12`的Transformer前接口；
7. CBAM reduction ratio 16、通道后空间顺序及`7 × 7`空间卷积；
8. P6-E00至P6-E03实验矩阵和执行顺序；
9. validation ROC AUC差异`0.005`的简化优先规则；
10. seed `42、2026、3407`及进入多seed确认的门槛；
11. 总参数不超过18,000,000、显存不超过8 GiB、有效batch 16；
12. 实施前必须恢复锁定train/validation产物并达到`59 passed`；
13. 阶段六禁止官方test、V2S、Grad-CAM++、外部数据集和阶段七实施；
14. 阶段六批准不包含依赖安装、commit或push。

本spec已于2026-07-23获得用户批准。后续实施必须继续遵守Gate顺序，不得把阶段六批准扩展到阶段七、官方test、依赖安装、commit或push。
