# 阶段七实验规格：EfficientNet-V2S升级、BiFPN迁移与最终核心模型

状态：**修订版已获用户批准并完成执行；Gate 0至Gate D全部通过**
创建日期：2026-07-24
批准日期：2026-07-24
独立V2S + CBAM修订批准日期：2026-07-24
适用起点：`phase-6-final`
预期起点提交：`3c10d2d2bc9139a15f262b33c259eb8d1d500230`

## 1. 本规格的执行地位

本文件依据以下权威材料起草：

1. `docs/research_plan_v2_post_baseline.md`；
2. `docs/phase6_bifpn_cbam_ablation_spec.md`；
3. `docs/phase6_progress_report.md`；
4. 阶段五冻结候选和阶段六冻结模块定义。

用户已于2026-07-24明确批准本spec，阶段七按本文Gate顺序实施。

在P7-E01正式训练开始前，已提出增加独立`V2S + CBAM`以形成完整V2S二因素消融的建议。用户在P7-E01完成后确认此前遗漏批准，并于2026-07-24明确批准该修订。因此本次扩展不是根据P7-E01结果临时搜索结构；P7-E01保持为修订前后完全相同的共同对照，不需要重训。

用户在起草本spec时明确要求：

- 阶段七统一使用Conda环境`surf`；
- 如阶段七确实需要新增依赖，只能安装到`surf`环境中。

本次起草spec不需要也不会安装任何依赖。后续仅在本spec获批、实施中确认现有环境无法满足要求时，才允许把必要依赖安装到`surf`，并记录包名、版本、安装命令和安装原因。不得修改系统Python或创建其他替代环境。

本spec获批不自动授权：

- 官方test manifest、官方test DICOM、阶段四test预测或test指标读取；
- 阶段八Grad-CAM++或外部验证；
- commit或push；
- 修改、覆盖或删除阶段四至阶段六冻结产物。

## 2. 阶段七要完成的任务

阶段七按照单变量、可归因的顺序完成四项任务：

1. **V2S backbone独立切换**
   在阶段五冻结的384输入、预处理、Transformer和训练协议上，把EfficientNet-B0单独替换为EfficientNet-V2S，测量backbone升级的贡献。

2. **BiFPN向V2S迁移**
   在V2S上接入阶段六冻结的BiFPN `C=128、R=1`，保持144个Transformer token，测量多尺度融合相对V2S单backbone的增量贡献。

3. **CBAM向V2S独立迁移**
   在不加入BiFPN时，把阶段六冻结的CBAM迁移到V2S，测量CBAM相对V2S单backbone的独立贡献。

4. **形成规定的最终核心方法**
   在V2S + BiFPN上加入阶段六冻结的CBAM，形成导师规定的：

   ```text
   EfficientNet-V2S + BiFPN + CBAM + Transformer
   ```

阶段七最终必须同时给出两类结论：

- **规定的最终核心方法**：V2S + BiFPN + CBAM + Transformer；
- **validation经验最佳模型**：按本spec预先声明的validation ROC AUC规则选出的配置。

两者可以不是同一个模型。若规定核心方法没有超过阶段五P5-E02或其他阶段七候选，必须如实报告。

## 3. 核心研究问题

阶段七回答以下四个可分别归因的问题：

1. 在其他条件冻结时，EfficientNet-V2S能否优于阶段五EfficientNet-B0对照？
2. 阶段六冻结的BiFPN `C128/R1`迁移到V2S后，能否相对V2S单backbone提高validation排序性能？
3. 阶段六冻结的CBAM单独迁移到V2S后，能否相对V2S单backbone提高validation排序性能？
4. 在V2S + BiFPN上加入冻结CBAM后，两个模块是否产生互补收益，并形成稳定的规定核心方法？

阶段六中BiFPN和CBAM在B0上的负结果不得被隐去，也不得预设它们在V2S上必然有效。

## 4. 阶段范围

### 4.1 本阶段允许开展

- 使用阶段五冻结的384 × 384乳房crop+letterbox预处理；
- 使用ImageNet预训练EfficientNet-V2S并全量微调；
- 审计并固定V2S的P2-P5端点；
- 运行V2S + Transformer独立backbone实验；
- 把阶段六冻结的BiFPN `C=128、R=1`迁移到V2S；
- 把阶段六冻结的CBAM单独迁移到V2S；
- 把阶段六冻结的CBAM迁移到V2S + BiFPN；
- 保持四尺度压缩后的Transformer token数为144；
- 进行配置门、单元测试、参数量、梯度、预训练权重、显存和数据隔离审计；
- 使用seed 42完成阶段七初筛；
- 按第12节规则进行多随机种子确认；
- 编写仅基于train/validation的阶段七进度与结论报告。

### 4.2 本阶段明确禁止

- EfficientNet-V2M/V2L、ConvNeXt、Swin或其他backbone；
- 重新搜索BiFPN通道宽度、重复次数、融合方式、卷积核或上下采样方式；
- 重新搜索CBAM位置、reduction ratio、空间卷积核或CBAM数量；
- 改变Transformer层数、头数、隐藏维度、FFN宽度、位置编码或聚合方式；
- 改变阶段五预处理、输入分辨率或随机增强；
- 使用阶段六结果继续调整学习率、loss、pos_weight、早停或scheduler；
- 使用官方test选择模型、epoch、阈值或任何超参数；
- Grad-CAM++、ROI定位、分割、多任务学习或CC/MLO多视图融合；
- INbreast、VinDr-Mammo或其他外部数据集；
- 覆盖、删除或改写已有阶段四、阶段五、阶段六产物；
- 未经明确批准的commit或push。

## 5. 冻结控制项

阶段七以下内容不得改变：

- CBIS-DDSM calcification-only + whole mammogram数据方案；
- CC/MLO视图作为独立样本；
- patient-level train/validation划分；
- `MALIGNANT=1`；
- `BENIGN`和`BENIGN_WITHOUT_CALLBACK=0`；
- train 986张、validation 241张；
- 阶段五确定性乳房crop+letterbox；
- 输入尺寸`384 × 384`；
- train和validation均不使用随机增强；
- Transformer维度`D=512`、4层、8头、FFN=2048；
- 二维正弦位置编码、mean pooling、LayerNorm和单logit输出；
- `BCEWithLogitsLoss(pos_weight=1.84)`；
- AdamW、初始学习率`1e-4`、weight decay `1e-4`；
- ReduceLROnPlateau、最大50 epoch、AMP和梯度裁剪`1.0`；
- 有效batch size固定为16；
- validation ROC AUC作为唯一主要选择指标；
- seed集合固定为`42、2026、3407`。

阶段七直接引用、不重新训练的主要对照为：

```text
P5-E02_seed42
EfficientNet-B0 + Transformer
输入：384 × 384
预处理：乳房crop+letterbox
随机增强：无
Validation ROC AUC：0.6995632875
Validation PR AUC：0.6162027464
```

阶段五三seed validation ROC AUC均值和样本标准差为：

```text
0.6846840397 ± 0.0149272049
```

阶段六冻结输入为：

```text
BiFPN通道宽度：128
BiFPN重复次数：R1
CBAM插入位置：Transformer前的512 × 12 × 12特征
CBAM reduction ratio：16
CBAM空间卷积核：7 × 7
CBAM顺序：通道注意力 → 空间注意力
```

## 6. 数据隔离与manifest门

阶段七训练入口只能解析并加载：

```text
train.csv SHA-256: 3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716
val.csv   SHA-256: 15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0
```

配置、CLI和内部函数均不得接受test路径。每次正式运行必须确认：

- train/validation患者交集为0；
- 每个split内sample_id唯一；
- train和validation均包含两个类别；
- validation transform完全确定性；
- validation预测恰好覆盖241个唯一sample_id；
- 预测sample_id和标签与锁定validation完全一致；
- 运行元数据明确记录`test_split_loaded=false`；
- Git忽略的联合`manifest.csv`不是阶段七运行前置条件。

当前接手worktree缺少Git忽略的`artifacts/data/train.csv`和`val.csv`，因此完整pytest实际为`78 passed, 2 failed`。Gate 0前只能以只读、可审计方式恢复或挂载上述两个锁定文件，不得重新划分数据，也不得为运行旧版`verify_handoff.py`而复制、恢复或解析包含test记录的联合manifest。

## 7. 执行环境与依赖规则

阶段七唯一允许的Python环境为：

```text
Conda环境：surf
Python：C:\nonsystem\DevTools\conda\envs\surf\python.exe
Python版本：3.11.15
PyTorch：2.9.1+cu128
torchvision：0.24.1+cu128
CUDA：12.8
```

现有torchvision已经提供`efficientnet_v2_s`、`EfficientNet_V2_S_Weights`及阶段七所需基础算子，因此当前预计不需要新增Python依赖。

若实施中发现确需新增依赖：

- 只能安装到`surf`；
- 不得修改base环境、系统Python或其他Conda环境；
- 安装前必须说明该依赖为何无法用现有库替代；
- 安装后记录包名、精确版本和环境差异；
- 重新运行完整pytest和阶段七专项测试；
- 不得通过降级PyTorch、torchvision、CUDA相关包来解决实现问题。

## 8. EfficientNet-V2S端点与预训练权重

### 8.1 本地架构审计结果

2026-07-24使用指定`surf`环境、torchvision 0.24.1和合成`1 × 3 × 384 × 384`输入完成只读审计。V2S共有8个`features` stage，阶段七固定使用每个目标分辨率下语义最深的以下端点：

| 层级 | torchvision端点 | 相对步长 | 384输入下形状 |
|---|---|---:|---|
| P2 | `features[2]` | 4 | `48 × 96 × 96` |
| P3 | `features[3]` | 8 | `64 × 48 × 48` |
| P4 | `features[5]` | 16 | `160 × 24 × 24` |
| P5 | `features[7]` | 32 | `1280 × 12 × 12` |

这些端点在本spec获批后即冻结，不得根据validation结果改成其他stage。

### 8.2 预训练权重

阶段七固定使用：

```text
torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
预期缓存文件名：efficientnet_v2_s-dd5fe13b.pth
```

起草本spec时，该权重尚未存在于本机torch hub缓存。用户批准本spec后，Gate 0已从PyTorch官方地址下载并通过torchvision内置哈希检查加载。权威记录为：

```text
文件名：efficientnet_v2_s-dd5fe13b.pth
文件大小：86,721,253 bytes
SHA-256：dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74
```

正式运行继续记录：

- torchvision权重枚举名称；
- 来源；
- 文件SHA-256；
- 文件大小；
- 严格加载结果；
- 加载前后抽查张量哈希或数值摘要。

不得使用随机初始化冒充预训练，也不得用非官方、来源不明的同名文件替代。显存烟雾测试可以使用`weights=None`，但正式训练必须成功加载并记录上述预训练权重。

## 9. 阶段七模型定义

### 9.1 P7-E01：V2S + Transformer

- 使用EfficientNet-V2S全部`features`作为backbone并全量微调；
- 使用P5的`1280 × 12 × 12`输出；
- 通过阶段五同定义的`1 × 1 Conv2d(1280→512)`投影；
- 展平为144个512维token；
- 加入冻结的二维正弦位置编码；
- 输入冻结的4层Transformer；
- mean pooling + LayerNorm + 单logit分类。

P7-E01只改变backbone，用于回答V2S相对阶段五B0的独立贡献。

### 9.2 P7-E02：V2S + BiFPN + Transformer

P7-E02固定使用阶段六BiFPN定义：

- P2-P5分别经`1 × 1 Conv2d → BatchNorm2d → SiLU`投影到128通道；
- 非负快速归一化融合权重，`epsilon=1e-4`；
- 1次完整自顶向下和自底向上融合，即`R=1`；
- nearest-neighbor上采样；
- `3 × 3 MaxPool2d(stride=2, padding=1)`下采样；
- 融合后使用阶段六冻结的深度可分离卷积；
- 不增加新的残差、注意力或额外尺度。

四尺度输出分别使用kernel/stride为`8、4、2、1`的确定性`AvgPool2d`压缩到`12 × 12`，按P2、P3、P4、P5顺序沿通道拼接为`512 × 12 × 12`，再经：

```text
1 × 1 Conv2d(512→512, bias=false) → BatchNorm2d → SiLU
```

进行尺度混合，最终仍为144个Transformer token。

### 9.3 P7-E03：V2S + CBAM + Transformer

P7-E03使用与P7-E01完全相同的V2S P5投影，在`1 × 1 Conv2d(1280→512)`之后的`512 × 12 × 12`特征上加入阶段六冻结CBAM。P7-E03不包含BiFPN，用于测量CBAM在V2S上的独立贡献。

### 9.4 P7-E04：V2S + BiFPN + CBAM + Transformer

P7-E04在P7-E02已经固定的`512 × 12 × 12`尺度混合特征后加入阶段六冻结CBAM。

P7-E03和P7-E04使用完全相同的CBAM：

- 通道注意力使用全局平均和最大分支；
- 两分支共享`512→32→512`的1 × 1卷积MLP；
- reduction ratio固定为16；
- 通道注意力后接空间注意力；
- 空间注意力使用通道均值、通道最大值和`7 × 7 Conv2d(2→1)`；
- 不增加残差、BatchNorm、dropout或第二个CBAM；
- CBAM输出再展平为144个token。

P7-E04是导师规定的最终核心方法。其规定地位不等于validation经验最佳。

## 10. 参数量审计与硬上限

使用`surf`环境、torchvision 0.24.1和阶段六冻结模块定义得到以下起草期参数量估计：

| 候选 | 预计总参数 |
|---|---:|
| P7-E01 V2S | 33,444,433 |
| P7-E02 V2S + BiFPN R1 | 33,358,175 |
| P7-E03 V2S + CBAM | 33,477,843 |
| P7-E04 V2S + BiFPN R1 + CBAM | 33,391,585 |

Gate A必须用正式阶段七实现重新计算并记录总参数和可训练参数。阶段七硬上限固定为：

```text
36,000,000
```

超过上限必须停止，不得通过临时缩小Transformer、冻结backbone或改变BiFPN宽度来规避。

## 11. 实验矩阵与执行顺序

| 实验ID | Backbone | BiFPN | CBAM | Token数 | Seed | 目的 |
|---|---|---|---|---:|---:|---|
| P7-E00 | B0 | 否 | 否 | 144 | 42 | 引用P5-E02，不重训 |
| P7-E01 | V2S | 否 | 否 | 144 | 42 | V2S backbone独立贡献 |
| P7-E02 | V2S | C128/R1 | 否 | 144 | 42 | BiFPN迁移增量 |
| P7-E03 | V2S | 否 | 是 | 144 | 42 | CBAM迁移独立贡献 |
| P7-E04 | V2S | C128/R1 | 是 | 144 | 42 | 模块交互与规定核心方法 |

执行顺序固定为：

1. 引用P7-E00；
2. 完成P7-E01；
3. 在不改变训练协议的情况下依次完成P7-E02和P7-E03；
4. 使用完全冻结的P7-E02 BiFPN和P7-E03 CBAM定义完成P7-E04；
5. 按第12节预先声明规则完成多seed确认。

不得因P7-E01、P7-E02或P7-E03出现负结果而跳过后续规定实验，也不得在看到前一个结果后修改下一个模型定义。

## 12. 指标与选择规则

### 12.1 主要指标

- validation ROC AUC。

### 12.2 次要指标

- validation PR AUC；
- validation BCE loss；
- best epoch；
- train/validation loss和ROC AUC差距；
- 训练时间；
- 峰值CUDA显存；
- 参数量；
- 单样本推理时间。

PR AUC、loss、阈值分类指标、速度和显存不得覆盖主要选择规则。

### 12.3 单步增量解释

阶段七按以下固定差值解释组件贡献：

```text
V2S backbone贡献 = P7-E01 - P7-E00
BiFPN迁移贡献   = P7-E02 - P7-E01
CBAM迁移贡献    = P7-E03 - P7-E01
BiFPN存在时CBAM增量 = P7-E04 - P7-E02
CBAM存在时BiFPN增量 = P7-E04 - P7-E03
```

同时报告P7-E02、P7-E03和P7-E04相对P7-E00的总差值。任何负差值均须保留。

### 12.4 seed 42经验候选

在P7-E00、P7-E01、P7-E02、P7-E03和P7-E04之间比较：

- ROC AUC差异达到或超过`0.005`时选择较高者；
- 差异小于`0.005`时优先选择总可训练参数更少者；
- 参数量相同时优先选择新增模块更少者；
- 仍相同时按`P7-E00 → P7-E01 → P7-E02 → P7-E03 → P7-E04`顺序选择。

该规则只决定validation经验候选，不改变P7-E04作为规定核心方法的身份。

### 12.5 多随机种子确认

Gate C全部完成后固定配置，使用seed `2026`和`3407`：

1. **P7-E04规定核心方法必须完成三seed确认**，无论seed 42是否优于对照；
2. 如果seed 42经验候选不是P7-E04，且该候选相对P7-E04或P7-E00至少高`0.005`，则该经验候选也完成三seed确认；
3. P7-E00直接引用阶段五已有三seed结果，不重新训练；
4. 不对未满足上述条件的其他阶段七候选追加seed。

最终报告必须给出：

- 每个确认模型在seed 42、2026、3407下的validation ROC AUC和PR AUC；
- 三seed均值和样本标准差；
- 相对P7-E00的同seed差值；
- P7-E04与阶段七经验候选的同seed差值；
- 规定核心方法和validation经验最佳模型的最终并列结论。

若阶段七所有V2S候选的三seed证据均未超过阶段五P5-E02，则validation经验最佳模型继续保留P5-E02。

## 13. 训练与checkpoint规则

所有阶段七正式训练固定使用：

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
- `early_stopping_min_delta`只影响耐心计数，不影响数值最佳checkpoint保存；
- 同一实验目录存在时默认拒绝运行；
- 禁止自动清空、覆盖或续写不匹配配置；
- 恢复训练必须验证配置哈希、Git HEAD、seed和checkpoint来源一致。

## 14. CUDA显存与batch规则

目标GPU为约12 GB显存的RTX 5070 Ti Laptop GPU。Gate B硬上限为：

```text
torch.cuda.max_memory_allocated <= 8 GiB
effective_batch_size = 16
```

烟雾测试顺序固定为：

1. 物理batch 8、梯度累积2；
2. 若OOM或超过8 GiB，改为物理batch 4、梯度累积4；
3. 仍失败则改为物理batch 2、梯度累积8；
4. 仍失败则改为物理batch 1、梯度累积16；
5. 物理batch 1仍失败时，该候选登记为Gate B失败并停止正式训练。

降低物理batch时不得同时修改学习率、模型结构、AMP或其他训练超参数。所有候选不要求使用相同物理batch，但有效batch必须始终为16。

显存烟雾测试：

- 只使用合成张量；
- 不加载任何数据split；
- 可以使用`weights=None`；
- 必须完成前向、反向和一次优化器更新；
- 不写正式checkpoint；
- 不声明validation指标；
- 记录`test_split_loaded=false`。

## 15. 失败、OOM和异常记录

每次正式或失败运行必须使用唯一目录并登记。失败记录至少包含：

- 实验ID、配置和配置SHA-256；
- Git HEAD、分支和dirty状态；
- seed和确定性设置；
- 预训练权重枚举及权重文件SHA-256；
- 物理batch、梯度累积和有效batch；
- 失败阶段、最后完成epoch或batch；
- `failure_type=oom`、`nan`、`pretrained_weight`或`exception`；
- 异常摘要；
- 峰值CUDA显存；
- 已生成产物的哈希；
- `test_split_loaded=false`；
- 指标字段留空，不得填入0或历史最佳值。

OOM、负结果和中断运行不得通过删除目录、修改registry历史行或覆盖输出隐藏。

## 16. 预计实施文件

本spec获批后，预计新增：

```text
configs/phase7/manifest_lock.yaml
configs/phase7/P7-E01_V2S_seed42.yaml
configs/phase7/P7-E02_V2S_BiFPN_R1_C128_seed42.yaml
configs/phase7/P7-E03_V2S_CBAM_seed42.yaml
configs/phase7/P7-E04_V2S_BiFPN_CBAM_seed42.yaml
src/cbis_ddsm/phase7_model.py
src/cbis_ddsm/phase7.py
scripts/audit_phase7_architecture.py
scripts/smoke_phase7_memory.py
scripts/run_phase7_experiment.py
tests/test_phase7_model.py
tests/test_phase7_training.py
docs/phase7_progress_report.md
```

多seed配置只在达到第12.5节条件后新增。

阶段七优先从`src/cbis_ddsm/phase6_model.py`复用已经测试的基础组件定义，但不修改阶段六配置约束，不把V2S塞入`Phase6ModelConfig`，也不改变阶段六既有模型行为。若实施中必须修改共享文件，应先停止并向用户说明原因、影响文件、回归风险和验证方案。

输出目录固定使用：

```text
artifacts/experiments/phase7/architecture_audit_v1.json（修订前记录，保留）
artifacts/experiments/phase7/memory_smoke_v1.json（修订前记录，保留）
artifacts/experiments/phase7/architecture_audit_v2.json
artifacts/experiments/phase7/memory_smoke_v2.json
artifacts/experiments/phase7/P7-E01_V2S_seed42/
artifacts/experiments/phase7/P7-E02_V2S_BiFPN_R1_C128_seed42/
artifacts/experiments/phase7/P7-E03_V2S_CBAM_seed42/
artifacts/experiments/phase7/P7-E04_V2S_BiFPN_CBAM_seed42/
artifacts/experiments/phase7/experiment_registry.csv
```

每个正式运行保存：

- 完整解析配置及其SHA-256；
- Git元数据；
- train/validation manifest哈希；
- 预训练权重来源和SHA-256；
- 架构审计结果；
- epoch级指标和TensorBoard event；
- `best.pt`和`last.pt`；
- 最佳validation逐样本预测；
- 参数量、训练时间和峰值显存；
- `outcome.json`；
- 所有关键产物SHA-256；
- `test_split_loaded=false`。

## 17. 验收Gate

### Gate 0：实施就绪

- 用户明确批准本spec和阶段七实施范围；
- 起点确认为`phase-6-final`提交`3c10d2d2bc9139a15f262b33c259eb8d1d500230`；
- 不在游离HEAD上直接开始正式实现；实施分支及分支操作另行按用户授权执行；
- 锁定train/validation manifest可只读访问且哈希一致；
- 不恢复、不解析联合test manifest；
- 完整pytest恢复为预期`80 passed`；
- `surf`环境版本和GPU信息重新记录；
- V2S预训练权重来源、SHA-256和严格加载通过审计；
- 阶段四冻结边界及阶段五、阶段六产物未被修改；
- 未读取官方test。

### Gate A：架构与单元测试

- P2-P5端点、通道和空间形状与第8节一致；
- P7-E01只改变backbone；
- P7-E02严格使用BiFPN `C128/R1`；
- P7-E03严格使用冻结CBAM且不包含BiFPN；
- P7-E04严格使用冻结BiFPN和CBAM；
- 四个候选Transformer前均为`512 × 12 × 12`；
- 四个候选token数均为144；
- 所有候选输出单logit；
- 总参数不超过36,000,000；
- 合成输入前向、反向和梯度均为有限值；
- 官方V2S预训练权重严格、可审计地加载；
- 阶段五和阶段六既有测试不回归；
- 配置门拒绝B0、其他V2S型号、R2、非128通道BiFPN及非冻结CBAM参数。

### Gate B：CUDA显存烟雾测试

- P7-E01、P7-E02、P7-E03和P7-E04均完成前向、反向和一次优化器更新；
- loss和梯度有限；
- 有效batch保持16；
- 峰值显存不超过8 GiB；
- 物理batch和累积步数冻结到正式配置；
- 不加载任何数据split；
- 不写checkpoint、不声明validation指标；
- 全部记录`test_split_loaded=false`。

### Gate C：seed 42受控实验

- 严格按P7-E01 → P7-E02 → P7-E03 → P7-E04顺序运行；
- 四个实验完成或保留合规失败记录；
- 每次预测覆盖241个唯一validation sample_id；
- sample_id集合和标签与锁定validation一致；
- 从逐样本预测独立重算ROC AUC和PR AUC一致；
- 按第12节只使用validation ROC AUC解释增量并选择候选；
- 负结果完整保留；
- 全部运行记录`test_split_loaded=false`。

### Gate D：多seed确认与报告

- P7-E04完成seed 42、2026、3407三seed确认；
- 满足第12.5节条件的非P7-E04经验候选完成三seed确认；
- 给出单seed、均值、样本标准差和同seed差值；
- 报告best epoch、loss、过拟合趋势、训练时间、参数量和显存；
- registry、配置、checkpoint、预测和哈希核验通过；
- 明确区分规定核心方法与validation经验最佳模型；
- 明确说明阶段六B0负结果与阶段七V2S迁移结果之间的关系；
- 阶段七报告经用户审核；
- 未读取或运行官方test。

## 18. 需要用户审核的条款

请逐项批准或提出修改：

1. 阶段七只处理V2S backbone切换、冻结BiFPN迁移、冻结CBAM独立迁移和最终组合；
2. P7-E01、P7-E02、P7-E03、P7-E04实验矩阵与固定执行顺序；
3. V2S P2-P5端点为`features[2]、[3]、[5]、[7]`；
4. 输入继续固定为384 × 384，Transformer token继续固定为144；
5. BiFPN继续固定`C=128、R=1`，不重新搜索；
6. CBAM继续固定Transformer前位置、reduction 16和`7 × 7`空间卷积；
7. 训练协议、有效batch 16和validation ROC AUC主要指标保持不变；
8. 参数硬上限36,000,000、CUDA显存硬上限8 GiB；
9. seed固定为`42、2026、3407`；
10. P7-E04规定核心方法必须完成三seed确认；
11. 非P7-E04经验候选只在满足第12.5节门槛时追加三seed；
12. 规定核心方法与validation经验最佳模型允许不同并必须并列报告；
13. Gate 0前只恢复锁定train/validation，不恢复或解析联合test manifest；
14. 正式训练必须使用可审计的官方V2S预训练权重；
15. 阶段七唯一环境为`surf`，必要的新依赖也只能安装到`surf`；
16. 阶段七禁止官方test、Grad-CAM++、外部数据集和阶段八实施；
17. 本spec批准不包含commit或push。

本spec及独立`V2S + CBAM`修订均已于2026-07-24获得用户明确批准。P7-E01完成后曾按用户要求暂停；用户完成修订审计并批准后，又明确授权按照本spec连续完成后续训练。Gate 0至Gate D现已全部完成，完整结果见`docs/phase7_progress_report.md`。

执行结果区分为：

- 规定核心方法：P7-E04，三seed validation ROC AUC为`0.6684445399 ± 0.0144141602`；
- validation经验最佳模型：P7-E01，三seed validation ROC AUC为`0.6968069874 ± 0.0252671825`；
- 阶段五P5-E02对照：`0.6846840397 ± 0.0149272049`。

因此P7-E01相对阶段五对照的三seed均值提高`0.0121229477`，而P7-E04相对阶段五对照降低`0.0162394998`。这些结论仅来自train/validation；所有正式运行均记录`test_split_loaded=false`。

本spec的批准从未包含官方test、阶段八、commit或push；阶段七完成也不自动扩展这些授权。用户随后于2026-07-25单独授权将阶段七当前成果提交并推送到`phase-7-final`，该发布授权仍不包含官方test或阶段八。
