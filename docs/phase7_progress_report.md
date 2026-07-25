# 阶段七实施进度报告

日期：2026-07-25
状态：Gate 0至Gate D全部完成；全部正式运行仅使用train/validation

## 一、审批与范围

用户已于2026-07-24批准`docs/phase7_efficientnet_v2s_migration_spec.md`。

初版批准矩阵为：

1. P7-E01：EfficientNet-V2S + Transformer；
2. P7-E02：EfficientNet-V2S + 冻结BiFPN C128/R1 + Transformer；
3. P7-E03：EfficientNet-V2S + 冻结BiFPN C128/R1 + 冻结CBAM + Transformer。

在P7-E01正式训练前，已提出补充独立`V2S + CBAM`分支的建议。P7-E01完成后，用户确认此前遗漏批准，并于2026-07-24明确批准修订。修订后矩阵为：

1. P7-E01：EfficientNet-V2S + Transformer，已完成且不重训；
2. P7-E02：EfficientNet-V2S + 冻结BiFPN C128/R1 + Transformer；
3. P7-E03：EfficientNet-V2S + 冻结CBAM + Transformer；
4. P7-E04：EfficientNet-V2S + 冻结BiFPN C128/R1 + 冻结CBAM + Transformer，规定核心方法。

该建议提出时间早于P7-E01结果，因此不是根据P7-E01指标临时扩展搜索空间。用户随后批准按照修订spec连续完成剩余训练；P7-E02、P7-E03、P7-E04及满足门槛的多随机种子确认现均已完成。

## 二、Gate 0

- 实施起点：`3c10d2d2bc9139a15f262b33c259eb8d1d500230`；
- 实施分支：`codex/phase-7-v2s`；
- 环境：`C:\nonsystem\DevTools\conda\envs\surf\python.exe`；
- 未安装新增依赖；
- 只恢复锁定的train和validation清单；
- 未恢复或解析联合manifest；
- 未读取官方test。

锁定清单：

```text
train.csv SHA-256: 3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716
val.csv   SHA-256: 15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0
```

恢复清单后，实施前完整pytest为：

```text
80 passed
```

## 三、V2S预训练权重

权重通过torchvision 0.24.1的`EfficientNet_V2_S_Weights.DEFAULT`从PyTorch官方地址获得，并通过torchvision内置哈希检查严格加载。

```text
文件名：efficientnet_v2_s-dd5fe13b.pth
文件大小：86,721,253 bytes
SHA-256：dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74
```

## 四、修订前Gate A：架构与测试

已完成：

- 独立的阶段七V2S配置门；
- V2S P2-P5端点提取；
- V2S单backbone、BiFPN迁移和BiFPN+CBAM模型；
- 仅允许train/validation的训练入口；
- 架构审计、CUDA显存烟雾测试和阶段七专项测试。

测试结果：

```text
阶段七专项测试：19 passed
完整pytest：99 passed
```

修订前v1架构审计确认：

| 候选 | 总参数 | Transformer前特征 | Token |
|---|---:|---|---:|
| P7-E01 V2S | 33,444,433 | 512 × 12 × 12 | 144 |
| P7-E02 V2S + BiFPN | 33,358,175 | 512 × 12 × 12 | 144 |
| P7-E03 V2S + BiFPN + CBAM | 33,391,585 | 512 × 12 × 12 | 144 |

P2-P5分别为：

```text
48 × 96 × 96
64 × 48 × 48
160 × 24 × 24
1280 × 12 × 12
```

预训练权重严格加载通过。架构审计没有加载数据split、没有写checkpoint、没有声明正式指标，记录`test_split_loaded=false`。

```text
artifacts/experiments/phase7/architecture_audit_v1.json
SHA-256: AED093D77E09496F7666E4AEF5F5C3B9765D2DB5D82117A5AA63BC593D598983
```

## 五、修订前Gate B：CUDA显存烟雾测试

全部候选使用物理batch 8、梯度累积2、有效batch 16完成前向、反向和一次优化器更新：

| 候选 | 峰值CUDA显存（MiB） | 8 GiB上限 |
|---|---:|---|
| P7-E01 V2S | 2345.53 | 通过 |
| P7-E02 V2S + BiFPN | 2648.81 | 通过 |
| P7-E03 V2S + BiFPN + CBAM | 2651.91 | 通过 |

所有loss和梯度均为有限值。烟雾测试没有加载数据split、没有加载预训练权重、没有写checkpoint、没有声明正式指标，记录`test_split_loaded=false`。

```text
artifacts/experiments/phase7/memory_smoke_v1.json
SHA-256: B341F8C52BBE5BC36BD01939B23FFAC2C313677BC4680D987F3264E768106E5E
```

## 六、修订扩展Gate A/B

独立`V2S + CBAM`修订实施后：

```text
阶段七专项测试：20 passed
完整pytest：100 passed
```

v2架构审计覆盖修订后的四个V2S候选：

| 候选 | 总参数 | Transformer前特征 | Token |
|---|---:|---|---:|
| P7-E01 V2S | 33,444,433 | 512 × 12 × 12 | 144 |
| P7-E02 V2S + BiFPN | 33,358,175 | 512 × 12 × 12 | 144 |
| P7-E03 V2S + CBAM | 33,477,843 | 512 × 12 × 12 | 144 |
| P7-E04 V2S + BiFPN + CBAM | 33,391,585 | 512 × 12 × 12 | 144 |

四个候选均低于36,000,000参数上限。P2-P5形状、单logit、144 token和官方V2S权重严格加载全部通过。

```text
artifacts/experiments/phase7/architecture_audit_v2.json
SHA-256: 3AB7EA6D1B49080535A7C824512CF40E50617E76610046AF05012F2AC8C3BCE9
```

v2 CUDA显存烟雾测试结果：

| 候选 | 峰值CUDA显存（MiB） | 8 GiB上限 |
|---|---:|---|
| P7-E01 V2S | 2345.53 | 通过 |
| P7-E02 V2S + BiFPN | 2648.81 | 通过 |
| P7-E03 V2S + CBAM | 2348.41 | 通过 |
| P7-E04 V2S + BiFPN + CBAM | 2651.91 | 通过 |

```text
artifacts/experiments/phase7/memory_smoke_v2.json
SHA-256: 5E40E0CB87742110798209C7DA3585D1B4912E34EF96C399BF49B6DF12FC76CE
```

v1架构和显存审计作为修订前记录保留，未覆盖。v2审计不加载任何数据split、不写checkpoint、不声明正式指标，全部记录`test_split_loaded=false`。

修订后的Gate C顺序为P7-E01 → P7-E02 → P7-E03 → P7-E04。P7-E01后曾按用户要求暂停；用户审计并批准修订后，又明确授权按照spec连续完成后续训练。实际运行严格遵守该顺序。

## 七、Gate C：P7-E01 V2S独立backbone实验

P7-E01按照批准的阶段七spec完成，配置为EfficientNet-V2S + Transformer、seed 42。该实验相对P5-E02只替换backbone，继续使用384输入、乳房crop+letterbox、无随机增强和冻结训练协议。

| 项目 | 结果 |
|---|---:|
| 完成epoch | 14 |
| 最佳epoch | 4 |
| Validation ROC AUC | 0.7179266896 |
| Validation PR AUC | 0.6056954281 |
| 最佳epoch validation loss | 1.0098830140 |
| 早停 | 是，连续10个epoch无有意义提升 |
| CUDA峰值显存 | 2603.58 MiB |
| 总参数量/可训练参数量 | 33,444,433 / 33,444,433 |
| test_split_loaded | false |

P7-E01相对P5-E02 seed42：

```text
ROC AUC差值：0.7179266896 - 0.6995632875 = +0.0183634021
```

该差值仅是seed 42结果；阶段七最终结论以Gate C完整矩阵和Gate D三随机种子统计为准。

独立核验确认：

- 最佳预测为241行和241个唯一sample_id；
- sample_id集合及顺序与锁定validation完全一致；
- 标签与锁定validation完全一致；
- 从逐样本预测重算ROC AUC为`0.717926689576174`；
- 从逐样本预测重算PR AUC为`0.6056954281153017`；
- `metrics.csv`共14行，数值最佳epoch为4；
- 运行元数据和registry均记录`test_split_loaded=false`。

权威产物SHA-256：

```text
best.pt:                  C59FF946B2F9732EE0684306E68C9490FCBD493BCD9BABB18A22C45E0DA784C8
last.pt:                  F40E1ADD6099905B383C8FCA90890886A1DDEA3FACFF61F1E1F76F660C05009F
val_predictions_best.csv: 9E23702D927A9CF695381F639A5CE5F687E9AF739735577AF27E21BB596A7D3A
metrics.csv:               B73EEA563C5D3044E9AA08CEF5B3886F2F841F23BB80A7C42C4226293FDE02D4
run_config.json:           11AFB7D5D944EEB9699C4248730970DFB6B982B86254FE40F4A0B00083DB7D4F
outcome.json:              F8A9665562F2F31AF05E584C392ABB1D01A7DFAD86D391A23F0C2925DBB80EEB
```

用户在P7-E01运行期间明确要求完成E01后不要开始P7-E02，因此当时确实暂停。用户完成修订审计并批准后，又明确授权按照spec连续训练，后续实验才开始。

## 八、Gate C：seed 42完整受控矩阵

所有实验均使用相同的384输入、乳房crop+letterbox、无随机增强、训练协议和validation选择规则。P7-E00直接引用阶段五P5-E02，不重训。

| 实验 | 模型 | 完成epoch | 最佳epoch | Validation ROC AUC | Validation PR AUC | 峰值显存（MiB） |
|---|---|---:|---:|---:|---:|---:|
| P7-E00 | B0 + Transformer | 13 | 3 | 0.6995632875 | 0.6162027464 | 1560.81 |
| P7-E01 | V2S + Transformer | 14 | 4 | 0.7179266896 | 0.6056954281 | 2603.58 |
| P7-E02 | V2S + BiFPN + Transformer | 14 | 4 | 0.6966280069 | 0.5540184094 | 2906.97 |
| P7-E03 | V2S + CBAM + Transformer | 22 | 12 | 0.6873210195 | 0.5987404971 | 2605.80 |
| P7-E04 | V2S + BiFPN + CBAM + Transformer | 14 | 4 | 0.6679553265 | 0.5554048313 | 2911.69 |

所有运行均正常早停，且`test_split_loaded=false`。

按spec第12.3节计算的seed 42增量为：

```text
V2S backbone贡献       = P7-E01 - P7-E00 = +0.0183634021
BiFPN迁移贡献          = P7-E02 - P7-E01 = -0.0212986827
CBAM迁移独立贡献       = P7-E03 - P7-E01 = -0.0306056701
BiFPN存在时CBAM增量    = P7-E04 - P7-E02 = -0.0286726804
CBAM存在时BiFPN增量    = P7-E04 - P7-E03 = -0.0193656930
P7-E02相对P7-E00       = -0.0029352806
P7-E03相对P7-E00       = -0.0122422680
P7-E04相对P7-E00       = -0.0316079611
```

因此seed 42经验候选为P7-E01。它相对P7-E04高`0.0499713631`，相对P7-E00高`0.0183634021`，均超过`0.005`门槛。Gate D必须同时确认P7-E04规定核心方法和P7-E01经验候选。

P7-E02、P7-E03和P7-E04的独立预测核验均确认：

- 241行、241个唯一sample_id；
- sample_id集合和顺序与锁定validation完全一致；
- 标签与锁定validation完全一致；
- 从逐样本预测重算的ROC AUC和PR AUC与记录精确一致；
- `metrics.csv`数值最佳epoch与`outcome.json`一致；
- 运行元数据和registry均记录`test_split_loaded=false`。

seed 42新增实验权威产物SHA-256：

```text
P7-E02
best.pt:                  B0C31710C90E2FAE79354EDE8DBC43882B5A47FE490672C7A7E83C2B7917C64A
last.pt:                  4BFB3B88C770174711B2460C2A41B5B586BDCA5F838068685C9677EFDA5E80CC
val_predictions_best.csv: 1A1C02F3585E055B811CB38F3E77E6D54B5323C45BC8C19E4CB5ABB5EF70B2DF
metrics.csv:               071C5E1C010969CEFD67D64356E0221D3EB056BB6AA60F60D68D7664FED60ED3
run_config.json:           87BDBFF9E05DBE84175D1059FC8537F3405B8EFAC7BE90308A91ECE1016B62B5
outcome.json:              82FB25FFC036FEBF0744737E735600EE71A050946822253CEDFB0D1F0DD01986

P7-E03
best.pt:                  73AD156ED90B91AF5F8B8E590AD6F331ACCCF8B38BF069A4DABFA6FE5A90A0DF
last.pt:                  C06986DDB88254FA3CE8D18D56271159FE29146376CFCCC8B947D97D7FC68D84
val_predictions_best.csv: D5947FD5F6F050A95C4C650A9246CBA850A0DB7176D571689ACA98E043F1E7D8
metrics.csv:               5231E848284EC52E626E6DC20DFD7BE81FE56BE07A45585F050EABCA8670F0F0
run_config.json:           E14195AF6D5F2D391AF10CDB05E2F02A44312FFAB23FF38EC5093E87B5AA1210
outcome.json:              0BE9E09AE1E1B245C4E905927D54B346FD53518FF22CEF3ED2CC371C278E5F8B

P7-E04
best.pt:                  CFFA435DC3DFFEBF8EF584DECEA7D06336873200465425021AF1AEFE901B14D8
last.pt:                  E74D3B6D6C1DBE75546E9B3CEEAFD366B283866A5FBFD1FDD260CCDE8D47B729
val_predictions_best.csv: 45D6E2894ED35B1E4DCF35BD8066B872A4B23BD8A794A6C0D66D50DF0B99EB6B
metrics.csv:               E80551F433752775356C85A9AA88676F161A649560DC42A0BE6AA8A9E5B1066D
run_config.json:           93B29DD9E18610F5721F5267849096F76A26D83BFAE5B33062610434634116EC
outcome.json:              A2B4A2F2110493622DE21703E80A254876661543B91D96232642000FB6DBCBC5
```

## 九、Gate D：三随机种子确认

### 9.1 P7-E01 validation经验候选

| Seed | 最佳epoch | ROC AUC | PR AUC | 相对同seed P5-E02 |
|---:|---:|---:|---:|---:|
| 42 | 4 | 0.7179266896 | 0.6056954281 | +0.0183634021 |
| 2026 | 7 | 0.7036798396 | 0.5897810954 | +0.0189003436 |
| 3407 | 13 | 0.6688144330 | 0.5742412438 | -0.0008949026 |
| 均值 | — | 0.6968069874 | 0.5899059224 | +0.0121229477 |
| 样本标准差 | — | 0.0252671825 | 0.0157274637 | — |

### 9.2 P7-E04规定核心方法

| Seed | 最佳epoch | ROC AUC | PR AUC | 相对同seed P5-E02 |
|---:|---:|---:|---:|---:|
| 42 | 4 | 0.6679553265 | 0.5554048313 | -0.0316079611 |
| 2026 | 5 | 0.6542812142 | 0.5466131346 | -0.0304982818 |
| 3407 | 15 | 0.6830970790 | 0.6417966508 | +0.0133877434 |
| 均值 | — | 0.6684445399 | 0.5812715389 | -0.0162394998 |
| 样本标准差 | — | 0.0144141602 | 0.0526002886 | — |

P7-E01三seed ROC AUC均值比P7-E04高`0.0283624475`。同seed差值分别为`+0.0499713631`、`+0.0493986254`和`-0.0142826460`。

所有四次新增多seed运行均通过与Gate C相同的241样本独立核验。权威产物SHA-256：

```text
P7-E04 seed2026
best.pt:                  3CE919ECEB6493546D6085885B043D15B1563A69C8E028C76BCE7FB7392073CC
last.pt:                  1B0D62F88C47159B0233A1B75F713C44E55D97C70949EA577DC3D1DB18E809E6
val_predictions_best.csv: 9C5C8C792DDF083F8881DFD0FE50A58B456B926444C4844CB833A21D31979F74
metrics.csv:               12744293DBB8F58B0D86B27D6B52E82E59AA629812F752FC97B81C6DEC599EA0
run_config.json:           DEA27C25F44D15CD01E5B5E74A05EAA1A615BBC5E23B96C32B7E5F0DFD3CB458
outcome.json:              E59573036775C430067871818E56337B12EA141424CBF2B34950AE45F2FC34FF

P7-E04 seed3407
best.pt:                  B23806B319756F4510D86CF99134DEF307AC2E4458E1D060082FA2A51E416917
last.pt:                  62BFFDB45FFCE901AB2A10914BBB6EED3D1EBEFAB102012D84F74E11A6852E70
val_predictions_best.csv: 1839405135AD0D09370C98EF3E1F6484F27E108111D13DC67EF335D40A0D0456
metrics.csv:               DF0584131F3CB650B517670A74B76BCC6351313637A27B610DA27CA0B41BC75D
run_config.json:           DB0C627D0CD0FC7998E7F3F9644384541A0A62AAB8E55690ADABCC8A85492086
outcome.json:              8544BA506926EE9EB56B2129F009B914599E58B4F7C90DB4B82CCDC0C081C86F

P7-E01 seed2026
best.pt:                  07007736FE3EEB6A1316932CDCF87FD636FE900FA3E5E63C7CA08CDF2A446282
last.pt:                  46A411C8D7E3BA089679BD87467C887C64A6C4D6AED8342878CF7659A3DDBCBC
val_predictions_best.csv: B8903A4CE4ED23376DF3EAB78F76A60DC3FEA98993788D07442FE262E7B96080
metrics.csv:               8E3F9E997C5EB7DA06AC025AB4426B5BE0B4D0CAA60EC77D8444FF5E630464AA
run_config.json:           10202B7F177FA2B5D3944DF6A01C8B6F633DC953448AD0109B60013234048A23
outcome.json:              69F8E9F99227BF46E643BE052AA50F738F38ADB4357C5F9DD8849E24CCD03BF5

P7-E01 seed3407
best.pt:                  C384A125846D890EF3C97CFBC833E8FA31CF8B56017D881667917F1E5CBE0ECF
last.pt:                  02D42633302DBA118C068B11786E24263695159FD98611F3CBF56BBD5085A91E
val_predictions_best.csv: 7E580CC4A1C2B482DEFDBB53225464628816AE6FA1260ADE3B380104B00F3DFC
metrics.csv:               58942B862B2A725B54F1680EFC06C27A6C2DDAD9F34703B8AFEFAC1F47A35845
run_config.json:           76F25C9476B2FE5BEB5FB066400C95DD154D960CBA1E13F569AF45D6EE68B304
outcome.json:              32F7CE372FDE95F554D3B9D9ECFD9D1D8FD832E68D11A1C82682B93DD2AF711B
```

最终实验registry共有8条完成记录，全部`test_split_loaded=false`：

```text
artifacts/experiments/phase7/experiment_registry.csv
SHA-256: D82C7358A77F2094F7C658496EBF3B9A8DB950D2611B450F8EF389D8D89EC5AE
```

## 十、阶段七最终结论

1. 单独将阶段五P5-E02的B0替换为V2S，在三seed均值上带来`+0.0121229477` ROC AUC，P7-E01成为新的validation经验最佳模型。
2. 在V2S上迁移BiFPN或CBAM均未在seed 42相对P7-E01产生收益；二者组合的P7-E04也明显低于P7-E01。
3. 导师规定核心方法仍是P7-E04，即`EfficientNet-V2S + BiFPN + CBAM + Transformer`；其三seed ROC AUC为`0.6684445399 ± 0.0144141602`。
4. validation经验最佳模型是P7-E01，即`EfficientNet-V2S + Transformer`；其三seed ROC AUC为`0.6968069874 ± 0.0252671825`。
5. P7-E01三seed均值比阶段五P5-E02高`0.0121229477`，但这仍是validation证据，不得表述为官方test性能提升。
6. 阶段七没有读取官方test，没有安装新增依赖，没有覆盖阶段四至阶段六冻结产物。

## 十一、Gate验收状态

| Gate | 状态 | 结论 |
|---|---|---|
| Gate 0 | 完成 | 环境、起点、manifest和数据隔离通过 |
| Gate A | 完成 | 四候选架构、参数、权重和100项完整测试通过 |
| Gate B | 完成 | 四候选CUDA显存烟雾测试低于8 GiB上限 |
| Gate C | 完成 | P7-E01至P7-E04 seed 42受控矩阵完成 |
| Gate D | 完成 | P7-E04与P7-E01三seed确认、统计与报告完成；最终pytest为100 passed in 10.93s |

阶段七实施已完成。用户于2026-07-25单独授权将当前成果提交并推送到`phase-7-final`；阶段八和官方test复用仍未授权。
