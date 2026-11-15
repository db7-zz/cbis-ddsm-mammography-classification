# V2S CNN-only 补实验完成报告

日期：2026-11-12
状态：Gate 0、A、B、C、D 全部完成；仅使用 train/validation。2026-11-12 训练完成时未提交或推送；用户于 2026-11-15 单独授权发布，见第 11 节。

## 1. 结论与指标边界

按照预先冻结的主要指标 validation ROC AUC，CNN-only 三种子均值为 **0.7040378007 ± 0.0036455833**，高于 V2S + Transformer 的 0.6968069874 ± 0.0252671825。Transformer 相对 CNN-only 的同种子 ROC AUC 差值为 +0.0098081329、+0.0007875143、−0.0322880871，均值 −0.0072308133，因此未观察到跨种子一致的 ROC AUC 收益。

在 Transformer 配置上增加 BiFPN/CBAM 组合，ROC AUC 同种子差值为 −0.0499713631、−0.0493986254、+0.0142826460，均值 −0.0283624475，也没有形成跨种子一致收益。这评价的是模块组合增量，不能分解为两个模块各自的三种子效应。

**PR AUC 的结论不同，必须保留：** Transformer 在三个种子的 PR AUC 均高于 CNN-only，均值提高 0.0153730588（0.5899059224 对 0.5745328636）。本次不能概括成“Transformer 在所有指标上无效”或“CNN-only 全面更优”；主要选模指标仍为预先声明的 ROC AUC，不因结果改变。

以下 ± 均为三个随机种子的样本标准差（ddof=1），不是置信区间；只有三个种子，不声明统计显著性。CNN-only 的投影、归一化和分类头也与 Transformer 配置不同，差值属于配置整体对照，不能严格归因于单独 encoder。

## 2. 审批、起点和数据纪律

- 用户于 2026-11-12 明确回复“批准，开始执行”，批准 `docs/phase9_cnn_only_control_spec.md` 的三种子连续训练及汇总。
- 分支：`codex/cnn-only-supplement`；起点：`d2b17b8ff184d06b7db6fe27ee9553f17136bde0`。训练时工作区 dirty 状态与完整运行源码/配置 SHA-256 均已保存；没有 commit 或 push。
- 固定任务为 CBIS-DDSM calcification-only + whole mammogram，CC/MLO 独立视图，MALIGNANT=1，其余两类良性标签=0；数据方案继续标记为“复现假设”。
- Train 986 张、validation 241 张，患者交集为 0；没有重新划分。
- 全部新训练和汇总均记录 `test_split_loaded=false`，新增官方 test 推理 pass 为 0。没有读取 test manifest、test DICOM、旧 test 预测或外部数据。
- 此补实验在历史 test 结果已知之后，由导师要求追加；不是新的未触碰测试验证。CNN-only 未做官方 test 评估，不能写作测试集最佳模型。
- 未安装依赖、未下载权重或数据，未覆盖阶段四至阶段八产物；未处理 presentations、publication_figures、reports、w1、w2、w3。

锁定清单：

```text
train.csv: 3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716
val.csv:   15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0
```

## 3. 架构和冻结训练协议

新增模型为 `EfficientNet-V2S features → AdaptiveAvgPool2d(1) → Flatten → Linear(1280,1)`。384×384 输入得到 1280×12×12 特征；无投影、LayerNorm、位置编码、Transformer、BiFPN、CBAM 或分类头 dropout。保留原 backbone 的 BatchNorm 与 stochastic depth，全部参数微调。

官方 ImageNet 预训练文件 SHA-256：`dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`；每次训练从缓存权重严格加载，不使用历史分类 checkpoint 初始化。

输入、DICOM LUT、百分位截断、乳房 crop+letterbox 和 ImageNet 归一化与 P7-E01 完全一致，无随机数据增强。AdamW lr=1e-4、weight_decay=1e-4；BCEWithLogitsLoss(pos_weight=1.84)；最大 50 epoch；物理 batch 8、累积 2、有效 batch 16；AMP、deterministic、梯度裁剪 1.0。ReduceLROnPlateau factor=0.5、patience=3、min_lr=1e-6；早停 patience=10、min_delta=0.001；数值最高 ROC AUC 决定 best.pt，min_delta 只影响早停。

可比预算指相同上限与选择/早停规则，不强制相同实际训练轮数或墙钟时间。所有种子按 42→2026→3407 串行完成，没有因负结果改协议、跳过种子或追加搜索。

## 4. 逐种子 validation 结果

所有 PR AUC 均指 average precision，取各模型 **ROC AUC 最佳 epoch** 的同一份预测，不是另外搜索出的 PR 最佳 epoch。

| 模型 | Seed | 最佳 epoch | 完成 epoch | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|
| CNN-only | 42 | 10 | 20 | 0.7081185567 | 0.5810997744 |
| CNN-only | 2026 | 5 | 15 | 0.7028923253 | 0.5812550524 |
| CNN-only | 3407 | 11 | 21 | 0.7011025200 | 0.5612437640 |
| CNN+Transformer | 42 | 4 | 14 | 0.7179266896 | 0.6056954281 |
| CNN+Transformer | 2026 | 7 | 17 | 0.7036798396 | 0.5897810954 |
| CNN+Transformer | 3407 | 13 | 16 | 0.6688144330 | 0.5742412438 |
| CNN+BiFPN+CBAM+Transformer | 42 | 4 | 14 | 0.6679553265 | 0.5554048313 |
| CNN+BiFPN+CBAM+Transformer | 2026 | 5 | 15 | 0.6542812142 | 0.5466131346 |
| CNN+BiFPN+CBAM+Transformer | 3407 | 15 | 25 | 0.6830970790 | 0.6417966508 |

## 5. 三种子汇总

| 模型 | ROC AUC 均值 ± 样本标准差 | PR AUC 均值 ± 样本标准差 |
|---|---:|---:|
| CNN-only | 0.7040378007 ± 0.0036455833 | 0.5745328636 ± 0.0115089597 |
| CNN+Transformer | 0.6968069874 ± 0.0252671825 | 0.5899059224 ± 0.0157274637 |
| CNN+BiFPN+CBAM+Transformer | 0.6684445399 ± 0.0144141602 | 0.5812715389 ± 0.0526002886 |

CNN-only 在这三组对照中的 ROC AUC 均值最高，本组三种子的标准差也更小；该描述不能推广为总体方差或临床泛化优势。阶段七 P7-E01 的“当时 validation 经验最佳”与 P7-E04 的“导师规定核心方法”均保留历史定义，不根据本次结果改写阶段八冻结评估。

| 同 seed 差值（前者减后者） | 42 | 2026 | 3407 | 均值 |
|---|---:|---:|---:|---:|
| CNN+Transformer minus CNN-only，ROC AUC | +0.0098081329 | +0.0007875143 | -0.0322880871 | -0.0072308133 |
| CNN+Transformer minus CNN-only，PR AUC | +0.0245956537 | +0.0085260430 | +0.0129974798 | +0.0153730588 |
| CNN+BiFPN+CBAM+Transformer minus CNN+Transformer，ROC AUC | -0.0499713631 | -0.0493986254 | +0.0142826460 | -0.0283624475 |
| CNN+BiFPN+CBAM+Transformer minus CNN+Transformer，PR AUC | -0.0502905968 | -0.0431679607 | +0.0675554070 | -0.0086343835 |

## 6. 新训练耗时、资源和过拟合

| Seed | 运行起止（北京时间） | epoch 耗时合计/分钟 | 墙钟/分钟 | 最佳 epoch train loss | 最佳 epoch val loss |
|---|---|---:|---:|---:|---:|
| 42 | 14:22:58–16:34:59 | 131.66 | 132.01 | 0.266024 | 1.693432 |
| 2026 | 16:35:13–18:22:16 | 106.74 | 107.05 | 0.431073 | 1.086051 |
| 3407 | 18:22:30–20:17:04 | 114.27 | 114.57 | 0.245157 | 1.886326 |

三次均正常早停，没有 OOM、NaN、失败重跑或中断。共 56 个正式 epoch。后期 train loss 持续降低而 validation loss 上升，三次末轮 validation loss 分别约 3.3922、3.1663、3.4330，存在明显过拟合；报告使用最佳 ROC checkpoint，不使用末轮 checkpoint 替代。每轮预处理回退 train=9、validation=1，与冻结流程一致。

CNN-only 总/可训练参数均为 20,178,769；P7-E01 为 33,444,433，减少约 39.66%。新训练三次 CUDA 峰值 allocated 显存均为 2188.90 MiB；合成 Gate B 为 2035.24 MiB，低于 8 GiB 上限。耗时会受 DICOM 预处理和系统负载影响，未做受控速度基准，不据此宣称模型速度优劣。

## 7. 来源、核验与回归

- 六次阶段七结果只读引用 `C:/Users/db_7/.codex/worktrees/117d/surf/artifacts/experiments/phase7/`。12 个 checkpoint/预测哈希与阶段七报告一致；训练预算、预处理配置、manifest 哈希一致，未重新训练已有模型。
- 阶段八提取前景函数的重构与旧版比较，32 个合成案例的像素、crop、回退与几何决策完全一致；Dataset 与 training.py 运行源码哈希未变。
- 每份新预测恰好包含 241 个唯一 sample_id，ID 顺序、标签与锁定 validation 一致；logit/probability 一致，ROC AUC/PR AUC 独立复算与 epoch 日志及 outcome 一致。
- 新模型全部 6 个 best.pt/last.pt 均在 CPU 严格加载，通过配置、seed、epoch、best metric、manifest 和有限张量核验；未执行额外模型推理。
- Gate 0/A/B：`artifacts/experiments/phase9/gates_v1.json`，PASS。新训练期间运行源码/配置未变化。
- 完整回归：实施后 `158 passed in 38.10s`；三次正式训练完成后 `158 passed in 28.59s`。
- 接手最初 137 passed / 2 failed 是本工作区缺少两个 seed 42 validation 预测文件；核验后补齐，恢复为 139 passed，未绕过测试。

## 8. 权威产物

```text
artifacts/experiments/phase9/gates_v1.json
artifacts/experiments/phase9/final_checkpoint_audit.json
artifacts/experiments/phase9/experiment_registry.csv
artifacts/experiments/phase9/P9-E01_V2S_CNN_only_seed{42,2026,3407}/
artifacts/experiments/phase9/console_logs/seed{42,2026,3407}.log
artifacts/experiments/phase9/comparison/controls_summary.json
artifacts/experiments/phase9/comparison/validation_comparison.csv
artifacts/experiments/phase9/comparison/three_seed_summary.csv
artifacts/experiments/phase9/comparison/validation_auc_comparison.png
```

每个新运行保留 best.pt、last.pt、metrics.csv、run_config.json、outcome.json、val_predictions_best.csv、verification.json 及 TensorBoard。controls_summary.json 保留九次运行关键文件哈希，最终 artifact_manifest.json 记录本次实现和产物哈希。

新模型关键哈希：

| Seed | best.pt SHA-256 | validation 预测 SHA-256 |
|---|---|---|
| 42 | `6564f42c71dba616e19d648817db2302031d164a1e9bfe7000c55cbd59b67241` | `6e63d6d16f8e349a9525d72929c9393d5be98f84280872acb7a60057e58a2531` |
| 2026 | `d52724ebca8d108c3bba325cab08f41488a97de22a445f8d8a45a0a008bf1a94` | `4b4bf508eb934632e39f558b7ea64896eac15c431875a5f2dd6436c5fce6396b` |
| 3407 | `7abc1e6b8cdc982e747aa086c8d88f35e1f221e29bb3e447f29ef5ced3252adf` | `e7f831145fb44a5acb398f20b04095bbcd908fcf99fcf80e84f3ad71cd2f6f74` |

## 9. 可用于论文的结果表述

在相同患者级训练/验证划分、384×384 输入、确定性预处理及可比训练预算下，V2S CNN-only、V2S+Transformer、V2S+BiFPN+CBAM+Transformer 的三种子验证 ROC AUC 分别为 0.70404±0.00365、0.69681±0.02527、0.66844±0.01441。增加 Transformer 的 ROC AUC 增益在不同种子间方向不一致，BiFPN/CBAM 组合也未带来一致 ROC AUC 收益。Transformer 的三种子 PR AUC 则均高于 CNN-only，显示不同排序指标存在取舍。以上为已知历史测试结果后的补充验证实验，不构成新增官方测试证据或单个模块的因果证明。

## 10. Gate 验收

| Gate | 状态 |
|---|---|
| 0 接手、授权及来源 | 完成 |
| A 架构、权重、预处理与测试 | 完成 |
| B 合成 CUDA 显存与梯度 | 完成 |
| C 三种子正式训练及逐样本核验 | 完成 |
| D 九次运行离线汇总、报告和最终回归 | 完成 |

补实验实施范围已全部完成。B0 CNN-only、官方 test、外部验证、依赖安装和提交/推送不在本次授权内，未执行。

## 11. 2026-11-15 发布授权与范围

用户明确要求“按照之前的规格把这次更新上传到github”，单独授权提交并推送这次已完成的阶段九成果。目标为规格中的 `codex/cnn-only-supplement` 分支；按明确的 17 个文件清单提交模型、训练与审计/汇总脚本、三种子配置、manifest lock、专项测试、规格、报告及交接记录。

发布前核验原 artifact_manifest.json 中 52 个文件，哈希全部一致；原清单 SHA-256 为 `211f64eb1b2e1af217575fa72f93feeb5d6fa08713289c8b5ede8fdd47870b74`。本次发布文档更新另行记录，原训练时源码、日志、checkpoint、预测和清单不修改。Git 忽略的 artifacts/backups/数据/checkpoint 及无关用户目录均留在本地。

2026-11-15 发布前完整回归为 `158 passed in 19.40s`，`git diff --check` 通过。此前两次沙箱运行被临时目录写权限限制阻止，未更改代码或测试绕过；随后使用新临时目录和获准的执行权限完成复核。Git 网络命令仅对本次连接临时禁用失效的本地代理，没有修改全局代理配置。

本次没有重新训练或运行官方 test，也没有安装依赖、合并 main、删除分支或处理用户文件。第 2 节和第 10 节中的未授权/未提交表述记录的是 2026-11-12 的实验完成状态。
