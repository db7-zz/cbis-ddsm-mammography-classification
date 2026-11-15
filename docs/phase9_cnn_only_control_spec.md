# 补实验规格：EfficientNet-V2S CNN-only 三随机种子对照

状态：用户已于 2026-11-12 明确回复“批准，开始执行”；Gate 0、A、B、C、D 全部完成。
批准范围：按本规格连续完成 V2S CNN-only 三种子 train/validation 训练、已有结果复用、测试、汇总及交接更新；不包含 test、依赖安装、commit 或 push。
创建日期：2026-11-12
编号：阶段九补实验（阶段八已完成，不覆盖既有阶段编号）。
实施起点：当前 HEAD `d2b17b8ff184d06b7db6fe27ee9553f17136bde0`。

后续发布授权：用户于 2026-11-15 明确要求“按照之前的规格把这次更新上传到github”，单独授权将本阶段成果提交并推送到 `codex/cnn-only-supplement`。第 8 节中的 commit/push 禁令是 2026-11-12 训练实施授权的历史边界，已由这次发布授权在限定范围内补充；官方 test、重训、依赖安装、main 合并和本地数据/checkpoint 上传仍不在授权内。

## 1. 来源、目标与解释边界

用户要求根据导师最新截图补做 CNN-only 实验。截图作为实验需求材料；执行范围和审批以用户指令及仓库 AGENTS.md 为准。

当前最关键的缺口是没有与 P7-E01、P7-E04 同协议的纯 CNN 基线。本补实验只新增 EfficientNet-V2S + 全局平均池化 + 单 logit 分类头，在 seed 42、2026、3407 下训练，复用核验后的阶段七结果回答：

1. V2S + Transformer 相对 CNN-only 是否在三个种子上有一致收益？
2. V2S + BiFPN + CBAM + Transformer 相对 V2S + Transformer 是否有一致收益？

第二个问题评价 BiFPN/CBAM 组合增量，不能分解成两个模块各自的三种子效应。现有 P7-E02、P7-E03 仅作为已完成的 seed 42 消融背景，不追加训练。

本次是已知历史 test 结果之后，由导师要求补充的 validation 对照。必须披露该时间顺序；结果不能称为新的未触碰测试验证，也不能根据历史 test 数字调整本补实验。

## 2. 接手审计事实

- 当前分支为 `codex/phase8-figures`，阶段八完成提交已存在；AGENTS.md、HANDOFF.md 和 handoff_state.yaml 仍以阶段七为最新状态，批准后需要补充当前状态，不抹去历史审批记录。
- 未跟踪目录为 `presentations/`、`publication_figures/`、`reports/`、`w1/`、`w2/`、`w3/`；本任务不删除、移动、编辑或暂存这些目录。
- 当前 train/validation 文件 SHA-256 与锁定值一致，样本数为 986/241，患者交集为 0。
- 原始阶段七产物位于 `C:/Users/db_7/.codex/worktrees/117d/surf/artifacts/experiments/phase7/`，本次只读核验。
- 六次既有训练的 run_config、outcome、metrics、最佳 validation 预测及 checkpoint 文件均已核验；两组模型的逐种子预测 ID、顺序、标签及重算 ROC AUC/PR AUC 一致，训练预算和预处理配置一致，均记录 `test_split_loaded=false`。
- 六次运行共 12 个 checkpoint/预测文件的 SHA-256 均已通过脚本核对，与受跟踪的阶段七报告一致；seed 42 的 checkpoint 和预测哈希还与阶段八 artifact_lock 对照通过。
- 为恢复现有回归测试，仅在本工作区补齐两份已锁定的 seed 42 validation 预测，使用禁止覆盖写入；未复制 test 文件或修改原实验目录。
- 只读审计保存于 `artifacts/experiments/cnn_only_supplement/readiness_20261112/existing_result_audit.json`。
- `dataset.py` 和 `training.py` 与阶段七运行时哈希一致。阶段八的 `preprocessing.py` 将 RGB→灰度→最大前景连通分量提取成 `breast_foreground_mask`，原 crop 算法保持一致；实施 Gate A 必须再验证其输出等价性并记录源码差异。
- 现有 `surf` 环境：Python 3.11.15、torch 2.9.1+cu128、torchvision 0.24.1+cu128，CUDA 可用，GPU 为 RTX 5070 Ti Laptop。
- 官方 V2S 预训练权重已缓存，本次不需要新增依赖或下载。
- 接手首轮完整 pytest 为 137 passed / 2 failed；两项失败均来自本工作区缺少上述 validation 预测，未改测试代码绕过。
- 补齐两份哈希锁定的 validation 预测后，完整回归测试为 `139 passed in 12.68s`，`git diff --check` 通过；此过程未读取官方 test 数据或预测。

## 3. 固定实验矩阵

| 标识 | 模型 | Seed | 动作 |
|---|---|---|---|
| P9-E01 | EfficientNet-V2S + GAP + Linear | 42、2026、3407 | 新增三次训练 |
| P7-E01 | EfficientNet-V2S + Transformer | 42、2026、3407 | 复用既有结果 |
| P7-E04 | EfficientNet-V2S + BiFPN + CBAM + Transformer | 42、2026、3407 | 复用既有结果 |

批准时已核验的 validation ROC AUC（CNN-only 当时待训练，完成结果见第 10 节）：

| 模型 | 42 | 2026 | 3407 | 均值 ± 样本标准差 |
|---|---:|---:|---:|---:|
| P9-E01 CNN-only | 待训练 | 待训练 | 待训练 | 待训练 |
| P7-E01 | 0.7179266896 | 0.7036798396 | 0.6688144330 | 0.6968069874 ± 0.0252671825 |
| P7-E04 | 0.6679553265 | 0.6542812142 | 0.6830970790 | 0.6684445399 ± 0.0144141602 |

不根据 seed 42 的结果决定是否执行另外两个种子；三个种子全部完成或保留真实失败记录。

## 4. CNN-only 定义

固定采用 torchvision `efficientnet_v2_s` 的全部 `features`，ImageNet 官方预训练，全量微调：

```text
RGB 384×384
→ EfficientNet-V2S features（1280×12×12）
→ AdaptiveAvgPool2d(1)
→ Flatten（1280维）
→ Linear(1280, 1, bias=True)
→ squeeze 为 batch 个 logit
```

移除原 ImageNet 分类器；不增加分类头 dropout、投影、LayerNorm、位置编码、Transformer、BiFPN 或 CBAM。保留 V2S features 自带的 BatchNorm 和 stochastic depth，与 P7-E01 backbone 一致。

这是导师要求的实用纯 CNN 配置对照。P7-E01 还包含 1280→512 投影、位置编码、Transformer 和 LayerNorm，因此比较只能解释为两种分类配置的整体差异，不能把全部差值严格归因于 Transformer encoder 单个模块。

预训练固定为 `EfficientNet_V2_S_Weights.DEFAULT`，文件 `efficientnet_v2_s-dd5fe13b.pth`，SHA-256 为 `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`。正式训练前验证哈希、features 严格加载和初始化来源，不加载任何既有分类 checkpoint 作为训练初始化。

## 5. 数据、预处理和训练预算

仅加载以下两个清单，禁止重新划分：

```text
train.csv: 3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716
val.csv:   15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0
```

沿用 calcification-only + whole mammogram（继续标记为复现假设）、独立 CC/MLO 视图和既有病理标签定义。所有 DICOM LUT、0.5/99.5 百分位截断、乳房 crop+letterbox 参数、ImageNet 归一化均与 P7-E01 一致；输入固定 384×384，无随机数据增强。

| 控制项 | 固定值 |
|---|---|
| Seed | 42 → 2026 → 3407，顺序串行执行 |
| 物理 batch / 梯度累积 / 有效 batch | 8 / 2 / 16，与既有训练一致 |
| 最大 epoch | 50 |
| Loss | BCEWithLogitsLoss(pos_weight=1.84) |
| 优化器 | AdamW，lr=1e-4，weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau，val_roc_auc/max，factor=0.5，patience=3，min_lr=1e-6 |
| 早停 | val_roc_auc/max，patience=10，min_delta=0.001 |
| 最佳 checkpoint | 数值最高 validation ROC AUC；min_delta 仅控制早停计数 |
| 其他 | AMP、deterministic=true、gradient_clip_norm=1.0 |
| DataLoader | workers=4、pin_memory、persistent_workers、prefetch_factor=2 |
| 日志 | epoch 指标、TensorBoard、最佳 validation 逐样本预测 |

可比预算指相同最大 epoch、早停、优化器和 batch 协议，不强制匹配实际早停 epoch 或墙钟时间。正式训练不因 CNN-only 更省显存而增大物理 batch；若 batch 8 烟雾测试失败，停止并报告，不自行改变 BatchNorm 条件。

## 6. 实施文件与隔离方式

批准后创建 `codex/cnn-only-supplement` 分支，保留当前工作区未跟踪用户文件。新增：

```text
configs/phase9/manifest_lock.yaml
configs/phase9/P9-E01_V2S_CNN_only_seed42.yaml
configs/phase9/P9-E01_V2S_CNN_only_seed2026.yaml
configs/phase9/P9-E01_V2S_CNN_only_seed3407.yaml
src/cbis_ddsm/phase9_model.py
src/cbis_ddsm/phase9.py
scripts/audit_phase9_cnn_only.py
scripts/run_phase9_experiment.py
scripts/summarize_phase9_controls.py
tests/test_phase9_model.py
tests/test_phase9_training.py
docs/phase9_progress_report.md
```

复用现有 `training.py`、Dataset 和确定性预处理，不放宽阶段七/八配置门，不改变既有模型行为。更新 AGENTS.md、docs/HANDOFF.md、docs/handoff_state.yaml 和 README.md 的当前状态与实验入口，明确新阶段授权范围；保留历史记录。若发现必须更改共享运行代码，先说明具体原因和回归验证，不临时扩大实验范围。

全部新运行写入 `artifacts/experiments/phase9/`，每个种子独立目录；已有目录拒绝覆盖。保存配置及源码哈希、Git HEAD/dirty、环境和权重来源、manifest 哈希、best.pt/last.pt、逐 epoch 指标、最佳 validation 预测、outcome、训练时间、显存、参数量和追加式 registry。训练及汇总均记录 `test_split_loaded=false`。

## 7. 验证与验收

1. Gate 0：批准本规格；记录新分支/HEAD，验证锁定 train/validation、六份复用产物的权威哈希及预训练权重，完整回归测试通过。原模型和产物保持原位，只读引用。
2. Gate A：验证 CNN-only 确无 Transformer/BiFPN/CBAM，384 输入、1280×12×12 特征和单 logit；合成前向/反向及梯度有限；features 权重加载正确；验证阶段八预处理重构前后等价；配置和 CLI 拒绝 test 输入、非预定 seed、超范围模型和不同训练预算。
3. Gate B：仅用合成张量完成 CUDA batch 8、累积 2 的前向、反向和优化器更新，峰值 allocated 显存不超过 8 GiB；不写正式 checkpoint。
4. Gate C：依次完成三个新种子；每次最佳预测覆盖恰好 241 个唯一 validation sample_id，顺序和标签一致；从 logit/probability 独立复算 ROC AUC、average precision，并核对数值最佳 epoch；不因负结果调参或额外重训。
5. Gate D：生成三模型逐 seed 及均值/样本标准差表、两个比较的同 seed AUC 差值、均值差值和差值方向计数，报告 PR AUC、best epoch、耗时、显存和参数量；完整回归测试和 diff 检查通过后更新交接文档。

均值和样本标准差使用三个种子，标准差 `ddof=1`。差值固定为 P7-E01−P9-E01、P7-E04−P7-E01，保留全部正负结果。“三个种子方向一致”仅为描述性证据，不等同于统计显著性或临床有效性；若方向不一致，明确报告未观察到跨种子一致收益，不以单 seed 或均值更高替代稳定性结论。

失败、OOM、非有限 loss 或中断须保留真实记录和唯一 attempt 目录，不删目录、不覆盖、不填造指标。协议不匹配或来源哈希不一致时先停止正式复用和训练，不用报告数字替代缺失预测。

## 8. 本次批准不包含的工作

- 读取或推理官方 test（包括 test manifest、DICOM、旧 test 预测）；不新增 test pass。
- 外部数据验证、Grad-CAM++、阈值调优、超参数搜索或模型集成。
- EfficientNet-B0 CNN-only；仅在论文还需要 B0 Transformer 贡献结论且用户另行批准时补做。
- 新依赖安装、权重/数据下载、覆盖历史产物、批量删除。
- commit、push、PR 或修改远端分支。

## 9. 已批准的具体执行范围

用户已批准：按照第 4–5 节冻结定义实施 V2S CNN-only，连续完成 seed 42、2026、3407 三次 train/validation 正式训练，复用核验后的 P7-E01/P7-E04 六次历史结果，并完成第 6–7 节实现、测试、汇总及交接更新。正常的逐种子运行不重复请求许可；范围之外的工作仍须单独授权。

## 10. 执行完成记录

三次新训练于 2026-11-12 按 42→2026→3407 完成，分别正常早停于 20、15、21 个 epoch，最佳 epoch 为 10、5、11。CNN-only validation ROC AUC 为 0.7081185567、0.7028923253、0.7011025200，均值/样本标准差为 **0.7040378007 ± 0.0036455833**；PR AUC 为 **0.5745328636 ± 0.0115089597**。

Transformer 相对 CNN-only 的同 seed ROC AUC 差值为 +0.0098081329、+0.0007875143、−0.0322880871，未显示跨种子一致 ROC AUC 收益；PR AUC 三个差值均为正，均值 +0.0153730588，必须同时报告。BiFPN/CBAM 组合相对 Transformer 的 ROC AUC 均值差为 −0.0283624475，方向同样不一致。

九次新/旧运行均已独立复算指标；六个新 checkpoint 严格加载通过；最终完整回归 `158 passed in 28.59s`。全部新运行 `test_split_loaded=false`，未新增官方 test 推理、未安装依赖、未提交或推送。完整结果与限制见 `docs/phase9_progress_report.md`。

2026-11-15 发布完成记录：用户另行授权后，阶段九实现提交 `de8a378faed92ce174ddc006016d32ffa1b6755d` 已推送到 `origin/codex/cnn-only-supplement` 并核验远端一致；发布前完整回归 `158 passed in 19.40s`。本段之前的“未提交或推送”描述为 2026-11-12 的实验完成快照。本次只发布代码、配置、测试和文档，不上传训练产物或数据。
