# 阶段八实验规格：冻结模型官方测试、Grad-CAM++解释与外部验证

状态：**用户已批准并完成Gate 0至Gate D；Gate D采用探索性BI-RADS任务迁移路线**
创建日期：2026-07-28
适用起点：已发布分支`phase-7-final`
预期起点提交：`a534698a68de836dc89340c5c52bd535289aca7e`

## 1. 本规格的来源与执行地位

本规格依据以下材料起草：

1. 导师对阶段六、阶段七工作的review；
2. `docs/research_plan_v2_post_baseline.md`；
3. `docs/phase6_bifpn_cbam_ablation_spec.md`与`docs/phase6_progress_report.md`；
4. `docs/phase7_efficientnet_v2s_migration_spec.md`与`docs/phase7_progress_report.md`；
5. 阶段五至阶段七已经冻结的validation选择结果。

导师review提出：

1. 对P7-E01执行一次官方测试评估，并把该结果作为摘要中的最终数字；
2. 对P7-E04执行一次官方测试评估，形成融合模型对照表，检查validation趋势能否推广到测试数据；
3. 生成Grad-CAM++结果，对比V2S、B0和V2S+BiFPN+CBAM；
4. 资源允许时，在INbreast上进行外部验证。

本文件只把上述意见转化为可审计的实验协议。用户已于2026-07-28明确批准执行Gate 0和Gate A；该批准不包含Gate B官方test、Gate C正式解释性运行、Gate D外部数据、依赖安装、commit或push。后续动作必须按第18节分别得到用户明确批准。

## 2. 阶段八的核心问题

阶段八只回答以下问题：

1. 已经完全基于validation冻结的P7-E01，在官方CBIS-DDSM test上的排序性能和阈值分类性能如何？
2. P7-E04相对P7-E01在validation上的劣势，是否也出现在同一官方test上？
3. P5-E02、P7-E01和P7-E04的Grad-CAM++关注区域有何差异，是否与钙化病灶标注一致？
4. 在不微调、不重选阈值的条件下，冻结模型能否迁移到INbreast或其他外部全视野数字乳腺摄影数据？

阶段八是**冻结模型评估与解释阶段**，不是新的模型选择、结构搜索或训练阶段。

## 3. 当前冻结事实

### 3.1 数据与任务

- CBIS-DDSM calcification-only + whole mammogram；
- CC/MLO视图作为独立样本；
- `MALIGNANT=1`；
- `BENIGN`与`BENIGN_WITHOUT_CALLBACK=0`；
- train 986张、validation 241张、official test 284张；
- train/validation/test患者两两无交集；
- 阶段五至阶段七模型选择只使用train/validation；
- 阶段四已经对官方test执行过一次基线推理，因此阶段八不是“首次未触碰测试集”的评估。

### 3.2 三个模型角色

| 角色 | 固定实验 | 架构 | 阶段八用途 |
|---|---|---|---|
| B0对照 | P5-E02 seed 42 | EfficientNet-B0 + Transformer | Grad-CAM++结构对照，不新增官方test推理 |
| validation经验最佳 | P7-E01 seed 42 | EfficientNet-V2S + Transformer | 官方test主结果、Grad-CAM++、外部验证 |
| 导师规定核心方法 | P7-E04 seed 42 | EfficientNet-V2S + BiFPN C128/R1 + CBAM + Transformer | 官方test对照、Grad-CAM++、外部验证 |

阶段八只使用上述seed 42数值最佳checkpoint。不得因为seed 2026或3407的validation指标不同而新增官方test推理、构造多seed ensemble或选择“测试表现最好”的seed。

### 3.3 阶段七validation结论

| 模型 | seed 42 ROC AUC | 三seed ROC AUC均值 ± 样本标准差 |
|---|---:|---:|
| P5-E02 | 0.6995632875 | 0.6846840397 ± 0.0149272049 |
| P7-E01 | 0.7179266896 | 0.6968069874 ± 0.0252671825 |
| P7-E04 | 0.6679553265 | 0.6684445399 ± 0.0144141602 |

P7-E01仍是validation经验最佳模型；P7-E04仍是导师规定核心方法。阶段八不得修改这两个定义。

## 4. 阶段范围

### 4.1 允许开展

- 对冻结的P7-E01 seed 42执行一次官方test推理；
- 对冻结的P7-E04 seed 42执行一次官方test推理；
- 为两个模型分别从其锁定validation预测冻结Youden J阈值；
- 生成ROC、PR、阈值分类指标、混淆矩阵和配对差值；
- 对P5-E02、P7-E01、P7-E04生成Grad-CAM++；
- 在能证明坐标一致时，用CBIS-DDSM ROI mask计算解释性定位指标；
- 在标签口径和访问许可审计通过后，对冻结模型进行INbreast外部验证；
- 编写阶段八报告，并明确区分validation、官方test和外部数据结论。

### 4.2 明确禁止

- 任何训练、微调、继续训练或外部数据适配；
- 根据官方test或INbreast结果修改模型、预处理、checkpoint、阈值或Grad-CAM++参数；
- 对P5-E02、P7-E02、P7-E03或其他seed新增官方test推理；
- test-time augmentation、模型集成、温度缩放或测试集校准；
- 把P7-E04在论文中的规定地位改写成validation经验最佳；
- 把P7-E01的validation提升表述成官方test提升；
- 把INbreast的BI-RADS标签未经审计直接等同于CBIS-DDSM病理标签；
- 以ROI crop或ROI mask作为分类模型输入；
- 阶段九模型改进、多视图融合、分割或新的backbone实验；
- 覆盖阶段四至阶段七任何正式产物；
- 未经批准的依赖安装、数据下载、commit或push。

## 5. Gate 0：实施前只读审计

阶段八实施前必须满足：

1. 从`phase-7-final`提交`a534698a68de836dc89340c5c52bd535289aca7e`创建独立实施分支，不在游离HEAD上开发；
2. 当前工作区无未授权改动，且不处理用户拥有的`w1/`或未跟踪报告；
3. 只读恢复或挂载锁定的`train.csv`与`val.csv`并验证：

   ```text
   train.csv SHA-256: 3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716
   val.csv   SHA-256: 15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0
   ```

4. 完整pytest恢复为预期`100 passed`；
5. 只读恢复并核验以下阶段七文件：

   ```text
   P7-E01 seed42 best.pt
   SHA-256: C59FF946B2F9732EE0684306E68C9490FCBD493BCD9BABB18A22C45E0DA784C8

   P7-E01 seed42 val_predictions_best.csv
   SHA-256: 9E23702D927A9CF695381F639A5CE5F687E9AF739735577AF27E21BB596A7D3A

   P7-E04 seed42 best.pt
   SHA-256: CFFA435DC3DFFEBF8EF584DECEA7D06336873200465425021AF1AEFE901B14D8

   P7-E04 seed42 val_predictions_best.csv
   SHA-256: 45D6E2894ED35B1E4DCF35BD8066B872A4B23BD8A794A6C0D66D50DF0B99EB6B
   ```

6. 为Grad-CAM++只读恢复P5-E02 seed 42的`best.pt`和validation预测；当前受Git跟踪的报告未给出该seed 42 checkpoint的权威SHA-256，必须从原始实验登记、运行元数据和文件本身恢复并新增可审计记录，禁止猜测或复制其他seed哈希；
7. 在获得官方test复用授权前，不恢复、不解析、不哈希test manifest、官方test描述CSV、test DICOM或ROI mask；
8. 在获得INbreast授权前，不下载、不解压、不解析外部数据。

若任一checkpoint、预测文件或来源无法验证，相关模型不得进入正式阶段八运行，必须把状态记录为`blocked_artifact_provenance`并向用户报告。

## 6. Gate A：阶段八代码、阈值冻结与测试

Gate A只允许使用合成数据和锁定validation产物，不读取official test或外部数据。

### 6.1 模型加载门

- P7-E01只能加载`P7-E01_V2S_seed42`配置和对应checkpoint；
- P7-E04只能加载`P7-E04_V2S_BiFPN_CBAM_seed42`配置和对应checkpoint；
- P5-E02只能用于Grad-CAM++，不得进入阶段八官方test入口；
- checkpoint必须严格加载，所有参数名、形状和数量一致；
- 记录配置哈希、checkpoint哈希、Git HEAD和dirty状态。

### 6.2 validation阈值冻结

P7-E01和P7-E04分别使用自己的241条锁定validation逐样本预测：

1. 验证sample_id集合、顺序和标签与`val.csv`一致；
2. 从logit重新计算概率、ROC AUC和PR AUC；
3. 验证结果与阶段七报告一致；
4. 使用`roc_curve(drop_intermediate=False)`；
5. 排除无限阈值；
6. 选择最大Youden J；
7. 完全平局时选择更高阈值；
8. 在任何official test文件可读前，分别写入不可覆盖的`threshold.json`。

两个模型不得共用阈值，也不得在看到test结果后重新冻结阈值。

### 6.3 测试要求

新增测试至少覆盖：

- 阶段八入口拒绝训练参数和非seed 42checkpoint；
- official test加载严格发生在两个阈值均冻结之后；
- P5-E02被official test入口拒绝；
- P7-E02、P7-E03、其他seed和ensemble被拒绝；
- 同一模型的official test输出目录存在时拒绝覆盖；
- Grad-CAM++输出尺寸、有限值、归一化和确定性；
- 三个模型的目标层严格匹配第8节；
- ROI坐标不一致时拒绝计算IoU/Dice；
- 外部标签未通过口径审计时拒绝生成“病理良恶性”指标；
- 阶段一至阶段七既有测试不回归。

Gate A完成后必须再次运行完整pytest。此时仍不得加载official test。

## 7. Gate B：两次受控official test推理

Gate B必须获得用户对“复用官方test”的单独明确授权后才能开始。

### 7.1 固定执行顺序

1. 验证两个模型的`threshold.json`已经冻结；
2. 一次性加载锁定official test manifest；
3. 验证284个唯一sample_id、151名患者、两个类别及train/validation/test患者隔离；
4. 对P7-E01 seed 42执行一个有序、无增强推理pass；
5. 完整写入P7-E01结果并关闭该输出目录；
6. 对P7-E04 seed 42执行一个有序、无增强推理pass；
7. 完整写入P7-E04结果并关闭该输出目录；
8. 基于已经保存的两份预测离线生成对照表，不再次运行模型。

若两次均完成，项目官方test模型推理pass总数从阶段四的1次增加为3次：

```text
阶段四基线：1
阶段八P7-E01：1
阶段八P7-E04：1
项目累计：3
```

任何失败重跑都必须先保留失败记录，并再次获得用户批准；不得把重跑隐藏为同一次推理。

### 7.2 指标

每个模型报告：

- ROC AUC；
- PR AUC（average precision）；
- sensitivity；
- specificity；
- accuracy；
- balanced accuracy；
- precision；
- F1；
- TN、FP、FN、TP；
- 使用各自validation冻结阈值的分类结果。

P7-E01的official test ROC AUC作为导师所述“摘要最终数字”的主要候选，PR AUC为重要次要指标。摘要仍必须同时注明数据集、样本单位、测试样本数和该评估不是首次未触碰test。

### 7.3 配对比较与不确定性

由于两个模型在完全相同的284张图像上评估：

- 使用患者级配对bootstrap计算P7-E01减P7-E04的ROC AUC和PR AUC差值；
- 每次以patient_id为单位重采样，并保留患者的全部视图；
- 固定bootstrap随机种子42；
- 生成2,000个包含两个类别的有效重采样；
- 报告百分位法95%置信区间；
- 不根据置信区间决定是否增加模型或测试pass。

### 7.4 融合对照表

正式报告至少包含：

| 模型 | seed 42 val ROC/PR | 三seed val ROC均值±标准差 | test ROC/PR | val→test变化 |
|---|---|---|---|---|
| P7-E01 V2S | 固定既有值 | 固定既有值 | 阶段八结果 | 描述性差值 |
| P7-E04 V2S+BiFPN+CBAM | 固定既有值 | 固定既有值 | 阶段八结果 | 描述性差值 |

该表用于检验趋势是否一致，不用于再次选择模型。P7-E02和P7-E03只保留validation消融结果，不新增official test数据。

## 8. Gate C：Grad-CAM++解释性比较

Gate C只有在checkpoint来源审计通过后才能运行。若使用official test图像或ROI mask，还必须先获得相应test复用授权。

### 8.1 固定模型与目标层

三模型都解释“恶性”单logit，禁用AMP并使用float32反向：

| 模型 | Grad-CAM++目标层 | 目标层输出 |
|---|---|---|
| P5-E02 B0 | `channel_projection` | `512 × 12 × 12` |
| P7-E01 V2S | `channel_projection` | `512 × 12 × 12` |
| P7-E04 V2S+BiFPN+CBAM | `token_compressor.mixer` | `512 × 12 × 12`，位于CBAM之前 |

三个目标层均位于Transformer之前，并保持同样的空间分辨率，避免用不同层级的特征图制造不公平比较。P7-E04的梯度仍通过后续CBAM与Transformer回传到融合特征。

Grad-CAM++优先使用项目内纯PyTorch实现，不新增依赖。若现有环境无法实现，必须先说明原因并单独申请依赖安装授权。

### 8.2 图像与热力图处理

- 使用与训练完全一致的384输入、乳房crop+letterbox和ImageNet归一化；
- 对每个模型、每张入选图像只针对恶性logit生成一张热力图；
- 对Grad-CAM++加权结果执行ReLU；
- 双线性上采样到`384 × 384`；
- letterbox padding不参与归一化或定位指标；
- 在有效乳房区域内执行每图min-max归一化；
- 若最大值等于最小值，记录`degenerate_heatmap`，不得伪造全零定位分数；
- 保存原图、单独热力图、叠加图和机器可读数组；
- 所有模型使用相同色图、透明度、分辨率和版式。

### 8.3 ROI mask与坐标审计

CBIS-DDSM的ROI mask只作为解释性真值，不作为分类输入。对同一whole-mammogram存在多个钙化异常时，合并为图像级union mask。

在计算定位指标前必须：

1. 从官方描述记录到whole-mammogram sample_id建立可追溯映射；
2. 验证ROI mask的原始尺寸、方向、laterality、view和空间坐标；
3. 对图像与mask应用完全相同的crop+letterbox几何变换；
4. 保存至少一批人工可视化叠加审计图；
5. 如果无法证明坐标一致，不得通过任意缩放强行对齐，也不得报告IoU/Dice。

坐标审计失败时，阶段八可保留Grad-CAM++定性图，但必须将定量定位状态标记为`not_comparable_roi_geometry`。

### 8.4 预先声明的定位指标

对有有效ROI mask的恶性图像：

- primary binary CAM mask：有效乳房区域内热度最高的10%像素；
- IoU；
- Dice；
- lesion hit rate：热力图全局最大点是否落入任一ROI；
- energy inside ROI：ROI内热度总和 / 有效乳房区域热度总和。

不得根据结果搜索5%、10%、15%等多个阈值后只报告最好者。10%是唯一正式二值阈值；energy inside ROI作为阈值无关补充。

统计以图像为基础，同时用patient_id聚类bootstrap给出95%置信区间。多ROI图像先合并mask，再计算一次图像级指标。

### 8.5 定性展示样本

正式主图固定展示12张，不根据模型热力图好坏人工挑选：

- 按`label × view`形成良性CC、良性MLO、恶性CC、恶性MLO四个层；
- 每层以`SHA-256(sample_id)`从小到大取前3张；
- 三模型并排展示；
- 若某层不足3张，报告实际数量，不从其他层补足。

可另做错误分析附图，但必须预先使用固定规则选取P7-E01的TP、TN、FP、FN，不能手工挑选“最好看”的案例。

## 9. Gate D：INbreast外部验证

Gate D是资源允许时的可选Gate，必须在用户确认数据可获得、许可允许且本地存储充足后单独批准。

INbreast原始论文报告115个病例、410张全视野数字乳腺摄影图像，包含肿块、钙化、结构扭曲和不对称，并提供专家勾画的XML轮廓：

- 原始论文：[INbreast: Toward a Full-field Digital Mammographic Database](https://recipp.ipp.pt/bitstreams/8184de8f-c540-4f48-9ab9-a4cb5d0851a8/download)

### 9.1 外部数据审计

模型运行前完成：

- 许可、下载来源、版本、文件数与总字节数记录；
- 原始文件SHA-256清单；
- patient/case/image/view/laterality唯一性检查；
- 图像可解码率和DICOM光度解释审计；
- 标签来源审计：病理、BI-RADS、病例级或病灶级；
- 钙化病例筛选规则；
- 多病灶、多标签和正常病例处理规则；
- patient-level隔离和样本单位说明；
- 与CBIS-DDSM标签口径的差异表。

若INbreast无法提供可验证的病理良恶性标签，不得把BI-RADS直接改写成CBIS-DDSM的`MALIGNANT/BENIGN`。此时只能：

1. 将分类结果标记为探索性的“BI-RADS任务迁移”；或
2. 放弃正式分类外部验证，只进行有轮廓病灶的解释性定位验证。

用户需要在审计报告后确认采用哪条路线。

### 9.2 冻结推理协议

- 只运行P7-E01 seed 42和P7-E04 seed 42；
- 不运行P5-E02外部分类，除非后续另行批准；
- 不训练、不微调、不做域适配；
- 使用CBIS-DDSM完全相同的384 crop+letterbox；
- ROC AUC和PR AUC为主要指标；
- 阈值分类指标使用各模型在CBIS-DDSM validation冻结的阈值；
- 不在INbreast上重新选择阈值；
- 报告所有纳入/排除病例及原因；
- patient级bootstrap报告95%置信区间；
- 结果单独标记为external，不与CBIS-DDSM test直接合并。

## 10. 推荐的其他数据集

阶段八优先使用INbreast。若需要在INbreast之外再选一个数据集，按当前项目目标推荐顺序如下：

| 优先级 | 数据集 | 适合原因 | 主要限制 | 建议角色 |
|---:|---|---|---|---|
| 1 | CMMD | 1,775名中国患者、3,728张乳腺摄影图像，良恶性由活检确认，与当前二分类标签最接近 | 不一定能严格形成calcification-only子集；单设备域和病例构成需审计 | INbreast之后的首选分类外部验证 |
| 2 | VinDr-Mammo | 5,000个四视图FFDM检查，双阅片、BI-RADS与病灶框，规模和定位标注较好 | 主要是影像学评估而非与CBIS完全同口径的病理二分类；PhysioNet限制访问 | 定位、域鲁棒性和多视图长期研究 |
| 3 | RSNA-SMBC | 接近20,000个筛查检查，来自两个站点，包含癌症/良性结局 | 数据量、算力和受控访问要求高；任务单位需从图像级重新冻结 | 大规模癌症外部验证 |
| 4 | EMBED | 大规模、多机构、包含钙化描述、病理与人口统计，可研究公平性 | 数据清洗、标签联结、许可和存储成本最高 | 长期规模化与公平性研究 |

权威入口：

- CMMD：[The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cmmd/)
- VinDr-Mammo：[PhysioNet v1.0.0](https://physionet.org/content/vindr-mammo/1.0.0/)
- RSNA-SMBC：[AWS Open Data Registry](https://registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection/)
- EMBED：[Emory HITI Open Data](https://github.com/Emory-HITI/EMBED_Open_Data)

如果阶段八只增加一个新外部数据集，选择INbreast；如果在INbreast之外再增加一个，优先选择CMMD。VinDr-Mammo更适合后续解释性/定位或多视图研究，不应在没有标签口径说明时取代CMMD的病理良恶性验证角色。

## 11. 输出目录与禁止覆盖

阶段八使用全新的输出目录：

```text
artifacts/experiments/phase8/gate0_artifact_audit.json
artifacts/experiments/phase8/thresholds/P7-E01/
artifacts/experiments/phase8/thresholds/P7-E04/
artifacts/experiments/phase8/official_test/P7-E01_seed42/
artifacts/experiments/phase8/official_test/P7-E04_seed42/
artifacts/experiments/phase8/official_test/paired_comparison/
artifacts/experiments/phase8/gradcam_pp/P5-E02_seed42/
artifacts/experiments/phase8/gradcam_pp/P7-E01_seed42/
artifacts/experiments/phase8/gradcam_pp/P7-E04_seed42/
artifacts/experiments/phase8/external/inbreast/
```

所有入口在目标目录非空时必须拒绝运行。不得自动删除、清空、续写或覆盖。失败运行保留唯一ID、异常摘要、已完成步骤和已生成文件哈希。

## 12. 正式产物

每个official test模型至少保存：

- 冻结配置与配置SHA-256；
- checkpoint来源和SHA-256；
- validation阈值与冻结记录；
- test manifest SHA-256；
- 推理开始记录和pass计数；
- 284条逐样本预测；
- metrics JSON；
- ROC、PR和混淆矩阵图；
- patient bootstrap区间；
- 运行环境、Git HEAD与dirty状态；
- `official_test_reused=true`；
- 关键文件SHA-256；
- outcome记录。

Grad-CAM++至少保存：

- 目标层和算法配置；
- 每样本原始热力图数组；
- 原图、热力图和叠加图；
- ROI映射审计；
- 每样本定位指标；
- 聚合统计和置信区间；
- 12张固定主图；
- 失败和退化热力图记录。

INbreast至少保存：

- 许可和来源记录；
- 原始文件清单哈希；
- 冻结外部manifest；
- 标签映射与排除规则；
- 逐样本预测；
- 指标、置信区间和数据集差异说明；
- `fine_tuned=false`与`threshold_refit=false`。

## 13. 批准后预计修改的文件

阶段八获批后预计新增：

```text
configs/phase8/artifact_lock.yaml
configs/phase8/official_test.yaml
configs/phase8/gradcam_pp.yaml
configs/phase8/inbreast_external.yaml
src/cbis_ddsm/phase8.py
src/cbis_ddsm/gradcam_pp.py
scripts/audit_phase8_artifacts.py
scripts/audit_phase8_gradcam.py
scripts/freeze_phase8_thresholds.py
scripts/run_phase8_official_test.py
scripts/run_phase8_gradcam_pp.py
scripts/audit_inbreast.py
scripts/run_phase8_external_validation.py
tests/test_phase8_evaluation.py
tests/test_gradcam_pp.py
tests/test_phase8_external.py
docs/phase8_progress_report.md
```

实际实施应尽量复用阶段四阈值与评估工具、阶段五预处理以及阶段七模型接口，不修改阶段五至阶段七既有模型行为。若需要修改共享文件，必须先说明原因、影响范围和回归验证计划。

## 14. Gate验收

### Gate 0：来源与环境

- 用户批准阶段八spec；
- 分支起点正确且不是游离HEAD；
- train/validation及三个模型产物来源可验证；
- 完整pytest为100 passed；
- 未读取official test或外部数据。

### Gate A：实现与冻结

- 阶段八专项测试通过；
- 两个模型validation指标可从预测精确复算；
- 两个阈值在test访问前冻结；
- Grad-CAM++合成测试、目标层和确定性通过；
- 外部标签门能拒绝未审计映射；
- 未读取official test。

### Gate B：官方测试

- 用户单独批准official test复用；
- P7-E01和P7-E04各完成且仅完成一个pass；
- 284个sample_id和标签完整；
- 两个模型指标、预测和哈希一致；
- 配对bootstrap和融合对照表完成；
- 无其他模型或seed的test推理。

### Gate C：解释性

- 三模型Grad-CAM++目标层一致可比；
- 主图样本按固定哈希规则选取；
- ROI坐标审计通过后才计算定位指标；
- IoU、Dice、hit rate和energy指标完整；
- 负结果、退化图和坐标失败未隐藏。

### Gate D：外部验证

- 用户单独批准INbreast数据访问和运行；
- 许可、哈希、样本与标签口径审计通过；
- 模型完全冻结，无微调和阈值重选；
- 外部指标、置信区间和限制完整；
- 若标签不等价，明确降级为探索性或仅定位验证。

## 15. 失败与停止条件

发生以下任一情况必须停止对应Gate：

- checkpoint或预测哈希不一致；
- 不能恢复P5-E02 seed 42的可信来源；
- train/validation/test患者隔离失败；
- test输出目录已经存在或pass计数不清楚；
- official test在阈值冻结前被加载；
- Grad-CAM++出现非有限梯度或无法证明ROI坐标一致；
- 外部数据许可不明确；
- INbreast标签不能支持声明的病理良恶性任务；
- 需要新增依赖但尚未得到授权；
- 实施需要超出本spec的新模型、训练或数据范围。

停止时保留错误证据，不删除失败输出，不用伪造的0填充指标。

## 16. 阶段八最终报告必须使用的措辞边界

- “P7-E01是validation经验最佳模型”；
- “P7-E04是导师规定核心方法”；
- “阶段八official test是模型冻结后的复用评估，不是首次未触碰test”；
- “P7-E01的阶段八test ROC AUC是该冻结模型在当前284张CBIS-DDSM官方test图像上的结果”；
- “P7-E04对照用于检验validation趋势是否延续，不用于再次选择或修改模型”；
- “Grad-CAM++提供模型关注区域证据，不证明因果机制或临床可靠性”；
- “外部数据结果受标签口径、设备、患者构成和样本量差异影响”。

## 17. 本阶段不会自动开展的后续工作

- 根据test或外部结果重新训练；
- 新增多视图融合；
- 用ROI监督训练定位模型；
- 使用CMMD、VinDr-Mammo、RSNA-SMBC或EMBED开展正式实验；
- 合并到`main`；
- commit、push或发布数据、checkpoint。

这些工作都需要新的spec或单独授权。

## 18. 需要用户逐项审核和批准

请逐项批准、拒绝或提出修改：

1. 阶段八只做冻结模型评估、Grad-CAM++和可选外部验证，不训练；
2. official test只运行P7-E01 seed 42和P7-E04 seed 42，各一次；
3. P7-E01 test ROC AUC作为摘要主要最终数字，PR AUC为重要次要数字；
4. 两模型分别使用各自validation冻结的Youden J阈值；
5. 使用患者级配对bootstrap比较P7-E01与P7-E04；
6. Grad-CAM++固定比较P5-E02、P7-E01和P7-E04；
7. 三模型目标层固定为Transformer前`512 × 12 × 12`特征；
8. 定位二值阈值固定为有效乳房区域热度最高10%；
9. ROI mask只用于解释性评价，坐标无法证明一致时不报告IoU/Dice；
10. 定性主图按`label × view`和sample_id哈希固定选12张；
11. INbreast先做许可、标签与坐标审计，再决定正式分类或仅定位验证；
12. INbreast不微调、不重选阈值；
13. INbreast之外优先推荐CMMD，但不纳入本阶段正式运行；
14. 阶段八实施、official test复用、外部数据访问和依赖安装分别授权；
15. 本spec批准不自动包含commit或push。

只有用户明确批准相应条款后，才能开始对应Gate。

批准记录：用户已于2026-07-28批准Gate 0与Gate A，随后明确指示“开始gateb”单独授权Gate B，并进一步明确指示“开始gate c”单独授权Gate C使用official test真实图像和ROI mask完成正式Grad-CAM++。用户于2026-07-29明确指示“开始gate d全套流程”并提供INbreast根目录。Gate D只读数据审计确认：统一图像表只有BI-RADS字段；隐藏工作表虽有55条非空`DG`诊断，但缺少图像连接键且与统一图像表同一行的BI-RADS有260处不一致，不能构造可靠的图像级病理标签。用户随后明确“选择路线1”，因此P7-E01和P7-E04已各完成一个`exploratory_birads_transfer`冻结外部推理pass；结果不作病理等价声明。Gate 0至Gate D结果见`docs/phase8_progress_report.md`。
