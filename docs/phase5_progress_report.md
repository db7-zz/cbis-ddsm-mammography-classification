# 阶段五实施进度报告

日期：2026-07-22
状态：Gate A至Gate D均已完成；阶段五验证集结论已批准发布

## 一、审批与执行边界

用户已于2026-07-21明确批准阶段五实施，并指定使用现有Conda `surf`环境。本次批准不包含安装依赖、读取官方测试集、开展阶段六架构工作、commit或push。

用户随后批准阶段五正式训练，并于2026-07-22提供8小时训练窗口，授权依次训练P5-E01、P5-E02；若时间明显充足则追加P5-E03。三个实验均在窗口内完整结束。

用户于2026-07-22再次明确要求立即启动下一步，P5-E04安全增强实验随后完整结束。

## 二、已经实现的内容

- 确定性选择最大的八连通乳房前景区域；
- 抑制仅位于图像外侧2%范围内的高占用扫描边框线；
- 在检测到的乳房前景四周增加2%安全边距；
- 对不满足有效性条件的图像执行完整原图回退，并记录具体原因；
- 保持宽高比进行缩放，并使用零值背景居中padding；
- 在其他模型架构参数不变的前提下，支持256、384和512三种输入尺寸；
- 实现梯度累积，使三种分辨率的有效batch size统一为16；
- 每个epoch记录预处理fallback数量、原因和CUDA峰值显存；
- 新增阶段五正式训练入口，只接受锁定的train/validation manifest，并使用禁止覆盖的独立实验目录。

本阶段没有安装任何新依赖。

## 三、Gate A：预处理审计

权威审计产物位于：

```text
artifacts/experiments/phase5/preprocessing_full_scan_v2
```

全量扫描结果：

- 锁定训练集：986张；
- 锁定验证集：241张；
- 扫描总数：1,227张；
- 加载的官方测试图像：0张；
- 正常执行crop和letterbox：1,217张；
- 明确回退到完整原图：10张；
- 10张图的回退原因均为`foreground_fraction_below_minimum`；
- 已人工检查22张覆盖审计图和全部10张fallback审计图；
- 10张fallback均为乳房在原图中占比较小的合理案例，并完整保留原始图像；
- 没有检测到乳房前景位于crop框之外的情况。

审计范围覆盖train/validation、CC/MLO、左右乳、良恶性，以及原始图像尺寸的最小值、中位范围和最大值。标签只用于保证审计覆盖，不参与crop位置或算法参数计算。

锁定manifest哈希：

- train：`3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716`
- validation：`15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0`

## 四、Gate B：CUDA显存烟雾测试

显存测试记录位于：

```text
artifacts/experiments/phase5/memory_smoke_v1.json
```

| 实验 | 输入尺寸 | 物理batch | 累积步数 | 有效batch | CUDA峰值显存（MiB） | 结果 |
|---|---:|---:|---:|---:|---:|---|
| P5-E01 | 256 | 16 | 1 | 16 | 1196.56 | 通过 |
| P5-E02 | 384 | 8 | 2 | 16 | 1425.41 | 通过 |
| P5-E03 | 512 | 4 | 4 | 16 | 1318.21 | 通过 |

每项烟雾测试均使用合成张量完成前向、反向和一次优化器更新，所有loss均为有限值。测试关闭了预训练权重，没有写入checkpoint，没有声明正式实验指标，也没有加载任何数据split。

## 五、Gate C：分辨率初筛

三个实验均使用seed 42、锁定的986张train和241张validation，只按validation ROC AUC选择分辨率。全部`outcome.json`均记录`test_split_loaded=false`。

| 实验 | 输入尺寸 | 完成epoch | 最佳epoch | Validation ROC AUC | Validation PR AUC | CUDA峰值显存（MiB） | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| P5-E00（冻结基线） | 256 | 12 | 12 | 0.658684 | 0.533763 | — | 仅引用，不重训 |
| P5-E01 | 256 | 17 | 7 | 0.669924 | 0.558237 | 1331.98 | 早停完成 |
| P5-E02 | 384 | 13 | 3 | **0.699563** | **0.616203** | 1560.81 | 早停完成，初筛最佳 |
| P5-E03 | 512 | 12 | 2 | 0.675580 | 0.565041 | 1454.08 | 早停完成 |

依据预先批准的唯一主要指标，P5-E02的384输入被选为Gate C最佳分辨率。它比P5-E03高0.023983，明显超过0.005的简化优先阈值；因此不存在需要用低分辨率平局规则处理的情况。P5-E04如获继续批准，应固定使用384输入。

独立核验确认每个实验的最佳预测均为241行、241个唯一sample_id，与锁定validation manifest完全一致；从逐样本预测重算的ROC AUC和PR AUC与训练记录一致。

P5-E03正式产物SHA-256：

- `best.pt`：`69E54DDA9A47AB7F30B73F340EAC9581230B3C15CE3CBAC32AA6F0EE1A63111E`
- `last.pt`：`CD40812630194568989512418BA3D92BD31BFDAE5F79859B4A10D418CAF5B12D`
- `val_predictions_best.csv`：`D035548A255BB858E331D28FFBA2509620C3229F460C456DF5EF02DE62F5A266`

P5-E01至P5-E03在实验登记表中均有完成记录：

```text
artifacts/experiments/phase5/experiment_registry.csv
```

## 六、P5-E04：安全增强

P5-E04固定Gate C选出的384输入和其余训练参数，只对train使用已批准的安全增强：组合仿射概率0.5（旋转±5°、双向平移不超过3%、缩放0.95–1.05）、gamma概率0.3（0.90–1.10）和对比度概率0.3（0.90–1.10）。validation不使用随机增强。

| 实验 | 增强 | 完成epoch | 最佳epoch | Validation ROC AUC | Validation PR AUC | CUDA峰值显存（MiB） |
|---|---|---:|---:|---:|---:|---:|
| P5-E02 | 无 | 13 | 3 | **0.699563** | **0.616203** | 1560.81 |
| P5-E04 | 安全增强 | 13 | 3 | 0.670139 | 0.588340 | 1560.81 |

P5-E04比P5-E02的validation ROC AUC低0.029424，PR AUC低0.027863。因此安全增强在seed 42下没有带来提升，最终候选保留P5-E02的“384输入、无增强”配置。

独立核验确认P5-E04最佳预测为241行、241个唯一sample_id，与锁定validation manifest完全一致；重算指标与训练记录一致，`test_split_loaded=false`。

P5-E04正式产物SHA-256：

- `best.pt`：`37B86122E81709BE453329A0BA84B5EA9B8239A70EBFF19063EDB72D4A91CFB6`
- `last.pt`：`70297AF41DB4E597C42DE7F3997B64BA86E1F62BBA6768A364661A2095585D90`
- `val_predictions_best.csv`：`EF7676D934EFE0FB2CE52CC8D8BD750A42C52900975EDE94A09F1FF4B41BEB1A`

## 七、实施验证

- 完整pytest：`59 passed in 6.66s`；
- 交接核验：`PASS`，55项检查通过，0项失败；
- `git diff --check`：通过，仅出现Windows LF转CRLF提示；
- 烟雾测试结束后没有残留Python进程或训练进程；
- 阶段四冻结产物未被修改。

## 八、Gate D：多随机种子确认

最终候选固定为384输入、无增强，seed 42引用P5-E02既有结果，并补充seed 2026和3407。

| Seed | 完成epoch | 最佳epoch | Validation ROC AUC | Validation PR AUC |
|---:|---:|---:|---:|---:|
| 42 | 13 | 3 | 0.699563 | 0.616203 |
| 2026 | 13 | 3 | 0.684779 | 0.580666 |
| 3407 | 15 | 5 | 0.669709 | 0.547121 |
| 均值 | — | — | **0.684684** | **0.581330** |
| 样本标准差 | — | — | **0.014927** | **0.034546** |

三个运行均早停完成，预测均为241个唯一validation sample_id，重算指标一致，且`test_split_loaded=false`。

seed 2026的`best.pt`/`last.pt`/预测SHA-256分别为`9F339A0C0B55D018C0B6C9C6706D74C36FB66ED538E45FB7E7DA3C5D4579A86D`、`6D95F1AFEFC324205DE4FEED968822CB2FFB3759FA11461C189E056698C8803A`、`43E2B6AA00D6540F40D66928E5B7C808DFB9E5641132973B0F54D81252C7ED46`。

seed 3407对应SHA-256分别为`1499B766F2B506539E418AE64214CF9540EAC6BDEF5C6EA99A593B8A01D976D1`、`49050A9CA5C19D1B3F1D301FD58F0853338EC5568C8BC7A9DF78FF5E0E56`、`45437A7B0E342BC6A73D0DF6CAD1F3344D58F1088CB1EF6FA62194A22F697E9A`。

## 九、下一步

Gate D已经完成，阶段五通过`codex/phase-5-experiments`实验分支发布。后续如进入阶段六，必须先单独编写并审批阶段六spec；不得自动运行官方测试。
