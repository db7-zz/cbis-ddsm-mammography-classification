# 阶段六实施进度报告

日期：2026-07-24
状态：Gate 0至Gate D已完成；CBAM多seed均值未提升，validation经验最佳模型保留阶段五候选

## 一、审批与执行边界

用户已于2026-07-23批准`docs/phase6_bifpn_cbam_ablation_spec.md`。本阶段只允许在阶段五最终候选上实施EfficientNet-B0的BiFPN、CBAM及组合消融。

本次已完成代码、配置、测试、架构审计、CUDA显存烟雾测试和P6-E01A正式训练。没有安装依赖，没有读取官方test，没有生成test指标，也没有commit或push。

## 二、Gate 0：实施就绪

- 已从提交`4e3b807421933ae3f4cb70bcce65263f2757e53c`创建阶段六工作分支，完成后更名为最终分支`phase-6-final`；
- 只恢复了锁定的`train.csv`和`val.csv`，没有复制或读取test manifest；
- train SHA-256：`3443C220D76FF8E50E63D17D3A6C46681E28B4FE2510EB56B41EE3682A595716`；
- validation SHA-256：`15E3611A765DB01B1765CFCD45067046F7B81E41EA827179E1A81F501DAE43D0`；
- 恢复manifest后，阶段五既有完整测试重新达到`59 passed`；
- 阶段六代码加入后，完整测试扩展为`80 passed`；
- 阶段四标签和冻结边界未被修改。

## 三、已经实现的内容

- 新增阶段六严格配置模型，拒绝384以外输入、V2S、非128通道BiFPN及非批准CBAM参数；
- 从EfficientNet-B0固定提取P2、P3、P4、P5；
- 实现非负快速归一化权重的BiFPN自顶向下与自底向上融合；
- 支持`C=128`、`R=1/R=2`；
- 将四尺度特征确定性池化到`12 × 12`并拼接、混合为512通道；
- Transformer token固定为144；
- 实现固定位置、reduction ratio 16、空间卷积`7 × 7`的CBAM；
- 新增仅允许train/validation的配置门和正式训练入口；
- 新增架构审计、显存烟雾测试和21项阶段六测试；
- 正式P6-E03配置已严格使用冻结的BiFPN C128/R1与固定CBAM定义生成，并在用户单独批准后完成训练。

## 四、确定性实现修正

第一次Gate B运行发现，PyTorch CUDA的`AdaptiveAvgPool2d`反向传播不支持严格确定性算法。该运行在第一个候选反向阶段中止，没有生成显存报告、checkpoint或实验指标。

为保持阶段六`deterministic=true`，实现改为：

- P2、P3、P4、P5分别使用kernel/stride为`8、4、2、1`的固定`AvgPool2d`，数学上等价地输出`12 × 12`；
- CBAM全局平均池化和最大池化分别使用显式`mean`与`amax`；
- 不使用`warn_only=True`绕过确定性要求。

修正后阶段六单元测试和CUDA反向均通过。初次生成的`architecture_audit_v1.json`属于修正前记录，权威架构审计为`architecture_audit_v2.json`。

## 五、Gate A：架构与参数审计

所有候选均确认：

- P2：`24 × 96 × 96`；
- P3：`40 × 48 × 48`；
- P4：`112 × 24 × 24`；
- P5：`1280 × 12 × 12`；
- Transformer前特征：`512 × 12 × 12`；
- token：`144 × 512`；
- 输出：单logit；
- 本地ImageNet-B0预训练权重可成功加载；
- 未加载任何数据split，`test_split_loaded=false`。

| 候选 | 总参数 | 18M上限 |
|---|---:|---|
| P6-E01A BiFPN R1 | 17,175,947 | 通过 |
| P6-E01B BiFPN R2 | 17,282,713 | 通过 |
| P6-E02 CBAM | 17,307,903 | 通过 |
| P6-E03候选 BiFPN R1 + CBAM | 17,209,357 | 通过 |
| P6-E03候选 BiFPN R2 + CBAM | 17,316,123 | 通过 |

权威产物：

```text
artifacts/experiments/phase6/architecture_audit_v2.json
SHA-256: 5E99C2252EF1B1721FD1F34C348458B8342D7EE2339FDEB200CA724B4DECC8B4
```

## 六、Gate B：CUDA显存烟雾测试

所有候选使用：

```text
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
输入: 384 × 384
物理batch: 8
梯度累积: 2
有效batch: 16
AMP: 开启
确定性算法: 开启
```

| 候选 | CUDA峰值显存（MiB） | 8 GiB上限 | 结果 |
|---|---:|---:|---|
| P6-E01A BiFPN R1 | 1729.38 | 8192 | 通过 |
| P6-E01B BiFPN R2 | 1960.44 | 8192 | 通过 |
| P6-E02 CBAM | 1427.41 | 8192 | 通过 |
| P6-E03候选 BiFPN R1 + CBAM | 1729.74 | 8192 | 通过 |
| P6-E03候选 BiFPN R2 + CBAM | 1965.04 | 8192 | 通过 |

所有loss和梯度均为有限值。烟雾测试没有加载数据split、没有加载预训练权重、没有写checkpoint，也没有声明正式指标，全部记录`test_split_loaded=false`。

权威产物：

```text
artifacts/experiments/phase6/memory_smoke_v1.json
SHA-256: 26AEA16EE1AB26CC32D97D634B1833FD90CD743D2414C5F9C0D36DC297564B7D
```

## 七、当前测试结果

- Python语法编译：通过；
- 阶段六专项测试：`21 passed`；
- 完整pytest：`80 passed`；
- 阶段五无模块控制与阶段五模型在相同权重和输入下输出严格相等；
- `git diff --check`将在Gate A/B文件范围冻结前再次执行。

## 八、Gate C进度：P6-E01A

P6-E01A按照用户授权单独运行，配置为EfficientNet-B0 + BiFPN C128/R1 + Transformer，seed 42。训练仅加载锁定的986张train和241张validation。

| 项目 | 结果 |
|---|---:|
| 完成epoch | 22 |
| 最佳epoch | 12 |
| Validation ROC AUC | 0.6669172394 |
| Validation PR AUC | 0.5351403331 |
| 最佳epoch validation loss | 2.2897244118 |
| 早停 | 是，连续10个epoch无有意义提升 |
| CUDA峰值显存 | 1863.60 MiB |
| 总epoch时间 | 5833.32秒（约97.2分钟） |
| test_split_loaded | false |

P6-E01A比冻结对照P6-E00的validation ROC AUC `0.6995632875`低`0.0326460481`。该差值只用于当前已完成实验的报告；BiFPN重复次数必须等P6-E01B完成后再按spec比较，当前不得提前冻结R1或跳过R2。

独立核验确认：

- 最佳预测为241行和241个唯一sample_id；
- sample_id集合与锁定validation完全一致；
- 逐样本标签与锁定validation一致；
- 重算ROC AUC为`0.6669172394043528`；
- 重算PR AUC为`0.535140333112728`；
- 运行配置和outcome均记录`test_split_loaded=false`；
- registry记录状态为`completed`。

权威产物SHA-256：

```text
best.pt:                  56F338A30C82AD5CD315F9F2F4AF1193D92E3FBAFBE79DB73CFC941F6C7D93ED
last.pt:                  CBD7A8EF6020FAF51E747350A08529C4BF841FD822DDEBDE1DA0DEED385E9EC6
val_predictions_best.csv: FAD62C11FAD82C7558FE34FE90C7BFB13F6076F6192220A7EE9B4366C45519E7
metrics.csv:               C741FD5C3330034E0120410FE7027358E75B379D6AAE29436B684E58C5E246E9
run_config.json:           8B6DE8DA0737086384342103481E01451A2A48639FEEED93E24A84F59BDB256D
outcome.json:              79914169B7279C066B3B65D7829D081F52C3535DC8904B5CB6E341AB70C81289
```

## 九、Gate C进度：P6-E01B与BiFPN冻结

P6-E01B按照用户单独授权运行，配置为EfficientNet-B0 + BiFPN C128/R2 + Transformer，seed 42。

| 项目 | 结果 |
|---|---:|
| 完成epoch | 22 |
| 最佳epoch | 12 |
| Validation ROC AUC | 0.6665950745 |
| Validation PR AUC | 0.5787494938 |
| 最佳epoch validation loss | 2.0418555572 |
| 早停 | 是，连续10个epoch无有意义提升 |
| CUDA峰值显存 | 2098.56 MiB |
| 总epoch时间 | 5644.00秒（约94.1分钟） |
| test_split_loaded | false |

独立核验确认预测为241行和241个唯一sample_id，ID集合及标签与锁定validation完全一致；重算ROC AUC和PR AUC与训练记录一致。

权威产物SHA-256：

```text
best.pt:                  D60AC060AEB32099E7932C45ECE106A4AF9FDA01F18711D36CEC3ACC453175F2
last.pt:                  D2DE9C4AC648DD62CF9BB0F75CE856A40479B89A5E90E392D6F1437BFC96A32B
val_predictions_best.csv: 10019B8A71DC3EC1C2AF9FCEF7874F8824C24BD7854F04AE47318EC8B9220DF1
metrics.csv:               54A8890368A4DA9E1F02F626D34C1F4C0980A6F8B0C384BB9FD796AC6B7B9FB4
run_config.json:           F9C2319A84C43B8FF67083F21018A9C8B70926F3FE39D4B44DAE3A3E508105EA
outcome.json:              3F70CD88000B01FDE464CE6023DA938866A898CABB554D4F6A86BF99B70C4795
```

BiFPN重复次数比较：

| 候选 | Validation ROC AUC | Validation PR AUC | 参数量 | 峰值显存（MiB） |
|---|---:|---:|---:|---:|
| R1（P6-E01A） | **0.6669172394** | 0.5351403331 | 17,175,947 | 1863.60 |
| R2（P6-E01B） | 0.6665950745 | **0.5787494938** | 17,282,713 | 2098.56 |

两者主要指标绝对差为`0.0003221649`，小于预先批准的`0.005`简化阈值。因此按照唯一主要指标和简化规则，正式冻结：

```text
BiFPN通道宽度：128
BiFPN重复次数：R1
冻结来源：P6-E01A_BiFPN_R1_C128_seed42
```

PR AUC更高的R2不能覆盖主要ROC AUC规则。机器可读冻结记录位于`configs/phase6/bifpn_selection.yaml`，SHA-256为`8D4350037848B6D1FE15735B664212008C80B03DE2A83DA4A5FEFE289DB31AD3`。

## 十、Gate C进度：P6-E02 CBAM独立消融

P6-E02按照用户单独授权运行，配置为EfficientNet-B0 + CBAM + Transformer，seed 42。CBAM固定插入在进入Transformer前的`512 × 12 × 12`特征上，通道压缩比例为16，空间卷积核为`7 × 7`。

| 项目 | 结果 |
|---|---:|
| 完成epoch | 13 |
| 最佳epoch | 3 |
| Validation ROC AUC | 0.7098725659 |
| Validation PR AUC | 0.6009777478 |
| 最佳epoch validation loss | 0.9038137571 |
| 早停 | 是，连续10个epoch无有意义提升 |
| CUDA峰值显存 | 1564.48 MiB |
| 总epoch时间 | 3360.36秒（约56.0分钟） |
| 总参数量/可训练参数量 | 17,307,903 / 17,307,903 |
| test_split_loaded | false |

独立核验确认：

- 最佳预测为241行和241个唯一sample_id；
- sample_id集合与顺序均与锁定validation完全一致；
- 标签与锁定validation完全一致；
- 从逐样本预测重算ROC AUC为`0.7098725658648339`；
- 从逐样本预测重算PR AUC为`0.6009777478441461`；
- `metrics.csv`共13行，最佳epoch和指标均与预测及`outcome.json`一致；
- 实验登记表记录状态为`completed`且`test_split_loaded=false`。

P6-E02比冻结对照P6-E00的validation ROC AUC `0.6995632875`高`0.0103092784`，比冻结的BiFPN R1候选高`0.0429553265`。该结果表明CBAM在seed 42下达到第11.5节的潜在多随机种子门槛，但阶段六seed 42选择必须等待P6-E03完成，当前不得提前启动多seed确认。

权威产物SHA-256：

```text
best.pt:                  6FACE7EF05707A5282BFD2B98FEE6C18E1682BF74D3B43DE384CA1246954787F
last.pt:                  05199949572482BF431AF3BED1895C9A5FA8F1266AB646E56AD2B88011ED58C5
val_predictions_best.csv: 3CE03CC58193759EEF2CBBF6B559F69EEA327FE007E306FDB5D783F8EB18CF75
metrics.csv:               D6E03E68FDBE802FA26B71963DB7941B9D93ED7DA299CCE9DECFE1C9B79EEB39
run_config.json:           2D7922F96655DFA2E1E7CDB16A8A468E38F2008E028AC01D6B0B1E2C12908F11
outcome.json:              F7B74174174702E92373991740FDA92FA45474A9AA4F2770E0E259742B19FC16
```

## 十一、Gate C进度：P6-E03 BiFPN与CBAM组合

P6-E03按照用户单独授权运行，配置为EfficientNet-B0 + 已冻结BiFPN C128/R1 + 固定CBAM + Transformer，seed 42。

| 项目 | 结果 |
|---|---:|
| 完成epoch | 16 |
| 最佳epoch | 6 |
| Validation ROC AUC | 0.6766537801 |
| Validation PR AUC | 0.5975603144 |
| 最佳epoch validation loss | 1.1515293601 |
| 早停 | 是，连续10个epoch无有意义提升 |
| CUDA峰值显存 | 1872.90 MiB |
| 总epoch时间 | 4185.81秒（约69.8分钟） |
| 总参数量/可训练参数量 | 17,209,357 / 17,209,357 |
| test_split_loaded | false |

独立核验确认：

- 最佳预测为241行和241个唯一sample_id；
- sample_id集合与顺序均与锁定validation完全一致；
- 标签与锁定validation完全一致；
- 从逐样本预测重算ROC AUC为`0.6766537800687286`；
- 从逐样本预测重算PR AUC为`0.5975603144316313`；
- `metrics.csv`共16行，最佳epoch和指标均与预测及`outcome.json`一致；
- 实验登记表记录状态为`completed`且`test_split_loaded=false`。

权威产物SHA-256：

```text
best.pt:                  93E68F45E62CBBBDA0703465935A40FDF6943932A2C74DD8D8D185492006B6C2
last.pt:                  DE7B472D631D37F2B93913D949C80F73A21D71F061651486B05F11E8C646CDAB
val_predictions_best.csv: 793C99E982CA6E78A77128D636113FA76E3DE7039CCE395876AE153A83296BE9
metrics.csv:               BD1F4DABC048A7ABAF58ABF46C6A1F1D929C5906617E4924AB219D7C1140E29E
run_config.json:           2B27BE30B773EC3B3A74BCF4AE8FDED374CB14335BDC297B2CB5F776D7B4566A
outcome.json:              0BE162A8C0B82BD84BD1147A04DC1A80BB6C42E0FF670A27A23E4F846794FF84
```

## 十二、Gate C结论与Gate D入口

按预先批准的唯一主要指标比较：

| seed 42候选 | 模块 | Validation ROC AUC | 相对P6-E00 |
|---|---|---:|---:|
| P6-E00 | 无新增模块 | 0.6995632875 | — |
| P6-E01（冻结R1） | BiFPN | 0.6669172394 | -0.0326460481 |
| P6-E02 | CBAM | **0.7098725659** | **+0.0103092784** |
| P6-E03 | BiFPN + CBAM | 0.6766537801 | -0.0229095074 |

P6-E02比次优的P6-E00高`0.0103092784`，达到`0.005`选择阈值，因此阶段六seed 42候选正式选择：

```text
P6-E02_CBAM_seed42
EfficientNet-B0 + CBAM + Transformer
```

P6-E03虽然比单独BiFPN R1高`0.0097365407`，但比P6-E02低`0.0332187858`，说明在本次B0实验中加入已冻结BiFPN没有与CBAM产生正向互补收益。负结果完整保留。

由于P6-E02相对P6-E00提升超过`0.005`，满足Gate D入口规则。Gate D固定P6-E02全部结构和训练配置，只将seed依次改为：

1. `P6-E02_CBAM_seed2026`；
2. `P6-E02_CBAM_seed3407`。

阶段五对应seed的冻结对照直接引用既有结果，不重新训练。用户随后于2026-07-24授权连续完成全部剩余Gate D训练；两个补充seed已依次完成，过程中仍只加载锁定的train/validation。

## 十三、Gate D：CBAM多随机种子确认

Gate D冻结配置SHA-256：

```text
P6-E02_CBAM_seed2026.yaml: 03BA4ABDF68F502BA7586AFCE1B999BD6C0D88C4A63060E8EB78BAEA5CBFDF8C
P6-E02_CBAM_seed3407.yaml: 0DBDCE8B60E6CFAD27CEBF891E49C81EA837132D83C364307037597E35062237
```

### 13.1 seed 2026

| 项目 | 结果 |
|---|---:|
| 完成epoch | 16 |
| 最佳epoch | 7 |
| Validation ROC AUC | 0.6590778923 |
| Validation PR AUC | 0.5342257712 |
| 最佳epoch validation loss | 1.5365556414 |
| 早停 | 是，连续10个epoch无有意义提升 |
| CUDA峰值显存 | 1564.48 MiB |
| 总epoch时间 | 4847.65秒（约80.8分钟） |
| test_split_loaded | false |

独立核验确认241个sample_id、顺序和标签与锁定validation完全一致，重算ROC AUC和PR AUC与训练记录一致。

权威产物SHA-256：

```text
best.pt:                  E281C0D7932C6A9B2CE098A66B7C47439427EB87CBD0A9217853872DE0E4BDAB
last.pt:                  760004F38B135B8308359ACF519F0B0B123DC76EA7702A04F956ED137D0C8AA4
val_predictions_best.csv: 12CDF073FBB22A75F6AF4E566809C9791CFB9892D159144210DD83488D9B60DD
metrics.csv:               786C5D41EEFFE754EC2136167D893A79007CDBAB245616E509C220D10095492D
run_config.json:           6F7EED2436F656B221F8D988AAF0C4AB61B62400A7ED5F9E075C9EA955D87379
outcome.json:              8956ECB3237CF6B0923EC430CEB319D6E431DA613829F7E4A802E7407D807E4D
```

### 13.2 seed 3407

| 项目 | 结果 |
|---|---:|
| 完成epoch | 27 |
| 最佳epoch | 17 |
| Validation ROC AUC | 0.6650558419 |
| Validation PR AUC | 0.5782228795 |
| 最佳epoch validation loss | 2.6950099775 |
| 早停 | 是，连续10个epoch无有意义提升 |
| CUDA峰值显存 | 1564.48 MiB |
| 总epoch时间 | 8277.27秒（约138.0分钟） |
| test_split_loaded | false |

独立核验确认241个sample_id、顺序和标签与锁定validation完全一致，重算ROC AUC和PR AUC与训练记录一致。

权威产物SHA-256：

```text
best.pt:                  34479B1BBBD5B703F139407489FBC96C2D83FCE2C226536A6D0967FF9F04F537
last.pt:                  B4BBC6AB208CEF590CBE7010DF6A33E1B3948DB5629C9F4F148C9D9BF3EBCDF1
val_predictions_best.csv: DF75ED393653CFACE701A1CAB87D7DAC58C876140D7C15AED5C846FF70294B31
metrics.csv:               3D8D76376946877F850112DB1109C0077EF767423756F1FC86343BC0E387B2E4
run_config.json:           FA400FF4D97D646C677EDCF6D46A166334ECC496BD97AAB06AC87438CAFF2353
outcome.json:              09284F9058ADE4680E2A070DC623D469AF8BA4056CF172033D2B36C472D1B8F1
```

### 13.3 三seed统计

阶段五对照的全精度指标只读取既有validation逐样本预测重新计算，没有重训阶段五，也没有读取test。

| Seed | 阶段五P5-E02 ROC AUC | 阶段六CBAM ROC AUC | 同seed差值 |
|---:|---:|---:|---:|
| 42 | 0.6995632875 | **0.7098725659** | +0.0103092784 |
| 2026 | **0.6847794960** | 0.6590778923 | -0.0257016037 |
| 3407 | **0.6697093356** | 0.6650558419 | -0.0046534937 |
| 均值 | **0.6846840397** | 0.6780021000 | **-0.0066819397** |
| 样本标准差 | **0.0149272049** | 0.0277620049 | — |

次要PR AUC统计：

| 统计 | 阶段五P5-E02 | 阶段六CBAM | 差值 |
|---|---:|---:|---:|
| 三seed均值 | **0.5813299100** | 0.5711421328 | -0.0101877772 |
| 样本标准差 | 0.0345455252 | **0.0339346332** | — |

## 十四、阶段六最终结论

seed 42的CBAM候选比阶段五对照高`0.0103092784`，但这一提升没有跨随机种子复现：

- seed 2026下降`0.0257016037`；
- seed 3407下降`0.0046534937`；
- 三seed ROC AUC均值下降`0.0066819397`；
- CBAM的ROC AUC样本标准差更高。

因此依据第11.5节的预先批准规则，阶段六validation经验最佳模型正式保留：

```text
阶段五P5-E02
EfficientNet-B0 + Transformer
384 × 384乳房crop+letterbox
无随机增强
```

阶段六冻结的结构研究结论为：

- BiFPN C128/R1与R2在B0上都低于阶段五对照，按简化规则冻结R1供后续迁移研究；
- CBAM在seed 42下有提升，但多seed均值低于阶段五对照，不能成为validation经验最佳模型；
- BiFPN R1 + CBAM组合没有产生正向互补收益；
- 负结果、checkpoint、预测、日志与哈希全部保留；
- 所有正式运行均记录`test_split_loaded=false`，没有使用官方test进行结构或参数选择。

这不改变导师规定的最终核心方法目标`EfficientNet-V2S + BiFPN + CBAM`。阶段六只完成B0模块消融；阶段七V2S迁移仍必须单独起草spec并获得用户批准，不得由本次训练授权自动启动。

最终完整pytest为`80 passed`。旧版`scripts/verify_handoff.py`仍要求联合`artifacts/data/manifest.csv`，而当前阶段六隔离worktree按数据纪律只恢复锁定的`train.csv`和`val.csv`，没有复制或解析包含官方test记录的联合manifest，因此该旧脚本在读取缺失文件时中止。阶段六没有为通过该脚本而扩大数据读取范围；其train/validation清单哈希、患者隔离、逐样本预测和`test_split_loaded=false`均已通过阶段六专用门与独立核验。

用户已于2026-07-24明确授权将阶段六代码、配置、测试和文档发布到最终分支`phase-6-final`，并删除其他阶段六相关分支。该发布不包含Git忽略的checkpoint、TensorBoard日志或本地实验目录，也不授权阶段七、官方test或依赖安装。
