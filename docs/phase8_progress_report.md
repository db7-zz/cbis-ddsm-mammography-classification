# 阶段八实施进度报告

日期：2026-07-29
状态：Gate 0至Gate D均已完成；Gate D采用探索性BI-RADS任务迁移路线

## 一、审批与执行边界

用户于2026-07-28批准开始执行阶段八Gate 0和“Gate 1”。本项目阶段八spec使用字母Gate，因此本次将“Gate 1”解释为Gate A。

用户随后于2026-07-28明确指示“开始gateb”，单独授权复用CBIS-DDSM官方test，对P7-E01 seed 42和P7-E04 seed 42各执行且仅执行一个推理pass。

用户随后于2026-07-28明确指示“开始gate c”，单独授权在official test真实图像和ROI mask上对P5-E02、P7-E01与P7-E04执行正式Grad-CAM++解释性比较。

用户于2026-07-29明确指示“开始gate d全套流程”，并指定INbreast数据根目录。该授权允许只读数据审计和后续符合spec的冻结模型外部验证，但不允许把BI-RADS未经审计直接改写为CBIS-DDSM病理标签；若标签不等价，仍须按spec在审计后确认探索性迁移或仅定位路线。

数据审计完成后，用户明确回复“选择路线1”，授权按预声明映射执行`exploratory_birads_transfer`。该结果只用于探索性跨数据集BI-RADS任务迁移，不声称与CBIS-DDSM病理标签等价。

截至Gate D完成，本阶段已执行：

- 从`phase-7-final`正确提交创建阶段八实施分支；
- 只读恢复并验证锁定train/validation和三个seed 42模型产物；
- 实现artifact lock、模型加载门、validation阈值冻结门和Grad-CAM++基础能力；
- 使用合成张量进行Grad-CAM++审计；
- 新增Gate B不可覆盖执行器、患者级配对bootstrap和专项测试；
- 验证阶段四冻结清单后，只读恢复哈希锁定的official test manifest；
- 对P7-E01和P7-E04按固定顺序分别完成一个无增强推理pass；
- 基于已保存预测离线生成指标、曲线、混淆矩阵和融合对照表；
- 审计326条official ROI记录到284张whole mammogram的坐标映射；
- 对125张固定图像和三个冻结模型生成375组正式Grad-CAM++；
- 对有效恶性图像计算固定top 10% IoU、Dice、lesion hit与ROI energy；
- 生成患者级bootstrap、配对比较、逐图机器数组和12张固定样本对照面板；
- 为INbreast建立独立授权记录、配置门、审计入口和13项专项测试；
- 对INbreast Release 1.0的1,622个文件生成SHA-256清单；
- 对410张INbreast DICOM完成全量像素解码、身份、视图、光度解释和有限值审计；
- 解析410行BI-RADS汇总、343份XML、343份OsiriX ROI与117份医疗报告；
- 冻结314张calcification-related CC/MLO候选图像和完整纳入/排除记录；
- 对P7-E01和P7-E04按固定顺序各执行且仅执行一个INbreast外部推理pass；
- 使用两个模型各自的CBIS-DDSM validation冻结阈值生成探索性分类指标；
- 生成患者级单模型置信区间、配对bootstrap与模型对照表；
- 运行完整pytest。

截至当前仍未执行：

- 对P5-E02、其他阶段七模型或其他seed执行新增official test推理；
- official test重跑、测试时增强、ensemble或测试集阈值重选；
- 与CBIS-DDSM病理标签等价的INbreast正式外部分类声明；
- INbreast XML解释性定位路线；
- 训练、微调、测试时增强或模型集成；
- 安装依赖；
- commit或push。

Gate 0/A的所有运行记录：

```text
official_test_authorized=false
official_test_loaded=false
test_split_loaded=false
```

Gate B的两个正式运行记录：

```text
official_test_authorized=true
test_split_loaded=true
test_inference_passes=1（每个模型）
training_performed=false
threshold_refit_on_test=false
test_time_augmentation=false
ensemble=false
```

Gate C正式运行记录：

```text
official_test_images_loaded=true
official_test_roi_masks_loaded=true
classification_test_inference_passes_added=0
formal_images=125
gradcam_image_model_pairs=375
training_performed=false
gate_d_loaded=false
```

## 二、分支与起点

实施分支：

```text
codex/phase-8-evaluation-xai
```

起点提交：

```text
a534698a68de836dc89340c5c52bd535289aca7e
```

该提交与本地及远端`phase-7-final`一致。阶段八没有在游离HEAD上继续开发。

## 三、Gate 0：锁定manifest恢复

只从既有阶段五/阶段七工作区复制明确列出的文件，没有复制联合`manifest.csv`或任何test文件。

| 文件 | SHA-256 | 只读 |
|---|---|---|
| `train.csv` | `3443C220D76FF8E50E63D17D3A6C46681E28B4FE2510EB56B41EE3682A595716` | 是 |
| `val.csv` | `15E3611A765DB01B1765CFCD45067046F7B81E41EA827179E1A81F501DAE43D0` | 是 |

恢复后、阶段八代码实施前的完整pytest为：

```text
100 passed in 13.30s
```

## 四、Gate 0：冻结模型产物审计

### 4.1 P5-E02 seed 42

阶段五跟踪文档此前没有记录P5-E02 seed 42的checkpoint和validation预测SHA-256。本次从原始实验目录、`run_config.json`、`outcome.json`和文件本身恢复并记录：

```text
best.pt:
5C99FDC049FE2ABD8BD33D45464656E5AED6BAC0CA3C93DA0D6A4D0C2AE2A41E

val_predictions_best.csv:
A07AADCDB2D1261A3E155D34DB7EB24C0ED3E06A72E5F8B0A0CA3FAB697D849E
```

核验结果：

- 实验ID：`P5-E02_seed42`；
- seed：42；
- best epoch：3；
- validation ROC AUC：`0.6995632875143184`；
- 参数量：17,274,493；
- checkpoint严格加载通过；
- `test_split_loaded=false`；
- 阶段八策略固定为`gradcam_only`，official test入口拒绝该模型。

### 4.2 P7-E01 seed 42

```text
best.pt:
C59FF946B2F9732EE0684306E68C9490FCBD493BCD9BABB18A22C45E0DA784C8

val_predictions_best.csv:
9E23702D927A9CF695381F639A5CE5F687E9AF739735577AF27E21BB596A7D3A
```

核验结果：

- best epoch：4；
- validation ROC AUC：`0.717926689576174`；
- validation PR AUC：`0.6056954281153017`；
- 参数量：33,444,433；
- checkpoint严格加载通过；
- 角色保持为validation经验最佳模型。

### 4.3 P7-E04 seed 42

```text
best.pt:
CFFA435DC3DFFEBF8EF584DECEA7D06336873200465425021AF1AEFE901B14D8

val_predictions_best.csv:
45D6E2894ED35B1E4DCF35BD8066B872A4B23BD8A794A6C0D66D50DF0B99EB6B
```

核验结果：

- best epoch：4；
- validation ROC AUC：`0.6679553264604811`；
- validation PR AUC：`0.5554048312928487`；
- 参数量：33,391,585；
- checkpoint严格加载通过；
- 角色保持为导师规定核心方法。

权威Gate 0审计：

```text
artifacts/experiments/phase8/gate0_artifact_audit.json
SHA-256:
56F157EB9269EDA16AF4EE8CFC6C1A65B6F553151F9B05335FA329A419308474
```

## 五、Gate A：实现内容

新增并锁定：

- 阶段八artifact lock；
- 只允许三个seed 42模型的配置门；
- 只允许P7-E01/P7-E04进入后续official test的模型白名单；
- official test项目累计pass预算`1 + 2 = 3`；
- checkpoint哈希、模型配置、seed、manifest哈希、best epoch和best metric严格加载门；
- validation逐样本ID、顺序、标签、logit/probability和指标复算门；
- 两模型各自的Youden J阈值冻结；
- 缺少单独授权JSON时拒绝test manifest加载；
- 纯PyTorch Grad-CAM++；
- 固定目标层、有效区域归一化、退化热力图标记；
- 固定top 10%定位mask及IoU、Dice、hit rate、ROI energy；
- 对额外模型、seed、训练、ensemble和TTA的配置拒绝。

Gate A完成时尚未实现或运行official test推理脚本；该入口仅在用户随后单独授权Gate B后新增。

## 六、Gate A：validation阈值冻结

两个阈值均在official test未授权、未恢复、未加载的状态下独立冻结。

### 6.1 P7-E01

```text
threshold: 0.657758296956317
Youden J: 0.35609965635738833
validation ROC AUC: 0.717926689576174
validation PR AUC: 0.6056954281153017
threshold.json SHA-256:
C6FF644AF73152145DD600EF0E3AB2894CB84934D7389225BA973E011D09A5FD
```

### 6.2 P7-E04

```text
threshold: 0.7634837635542924
Youden J: 0.3287514318442154
validation ROC AUC: 0.6679553264604811
validation PR AUC: 0.5554048312928487
threshold.json SHA-256:
8F2F6A1CA802332677B7EC356768A519688A56561499EEE02186345CCEE36762
```

阈值输出目录非空后拒绝再次冻结。两个模型不得共用阈值。

## 七、Gate A：Grad-CAM++合成审计

三模型均使用各自冻结checkpoint、float32和合成`1 × 3 × 384 × 384`输入完成前向与梯度计算：

| 模型 | 目标层 | 目标层形状 | 热力图形状 | 有限值 | CUDA峰值 |
|---|---|---|---|---|---:|
| P5-E02 | `channel_projection` | `1 × 512 × 12 × 12` | `1 × 384 × 384` | 是 | 404.07 MiB |
| P7-E01 | `channel_projection` | `1 × 512 × 12 × 12` | `1 × 384 × 384` | 是 | 650.22 MiB |
| P7-E04 | `token_compressor.mixer` | `1 × 512 × 12 × 12` | `1 × 384 × 384` | 是 | 705.31 MiB |

本次任意线性合成输入上，P7-E01与P7-E04的正类Grad-CAM++经ReLU后为退化零图。实现按spec记录为：

```text
degenerate_handling=recorded_no_localization_metrics
```

这不是正式图像解释结果，也不用于判断模型优劣。Gate C若获批准，将在真实、固定样本上继续记录每张图的退化状态；退化图不得伪造定位指标。

权威合成审计：

```text
artifacts/experiments/phase8/gradcam_pp_synthetic_audit.json
SHA-256:
DC09B31E364EA59C4401470144DDC97B2B0C7A7D7A8685DFF59A65E9C21858A8
```

## 八、Gate B：官方test冻结模型评估

### 8.1 授权、数据与执行纪律

Gate B授权记录：

```text
artifacts/experiments/phase8/official_test_authorization.json
SHA-256:
A282ECF32E0EEB9531E9F1EC7DB14DDE06AB433C0BF8DAC9455D90D55DEBD8AD
```

在恢复test manifest前先验证阶段四冻结清单：

```text
freeze_manifest.json SHA-256:
61176600394D4ABDFC62AAD075C9C2AA1567F9F03E654FF6117D89E6A867DE30

test.csv SHA-256:
AD631667905EC737F6D60EFFFD608EB6AAF280F3D2BC85864ED43DA1B56CF8A0
```

原始工作区、阶段四冻结备份和当前只读副本的`test.csv`哈希一致。正式预检确认：

- 284张whole-mammogram独立视图；
- 151名患者；
- 119张恶性、165张良性；
- 284个唯一sample_id；
- train、validation与test的patient_id和sample_id两两无交集；
- 两个模型使用相同的锁定384输入预处理；
- 两个模型各自使用Gate A已冻结的validation Youden J阈值；
- CUDA有序推理、无随机增强、无ensemble、无训练。

### 8.2 P7-E01：validation经验最佳模型

| 指标 | official test结果 |
|---|---:|
| ROC AUC | **0.7219760632** |
| PR AUC | 0.6122469838 |
| sensitivity | 0.7647058824 |
| specificity | 0.5696969697 |
| accuracy | 0.6514084507 |
| balanced accuracy | 0.6672014260 |
| precision | 0.5617283951 |
| F1 | 0.6476868327 |
| TN / FP / FN / TP | 94 / 71 / 28 / 91 |

使用阈值`0.657758296956317`。该ROC AUC是导师所述“摘要最终数字”的主要候选，但必须表述为：冻结P7-E01 seed 42在284张CBIS-DDSM官方test whole-mammogram视图上的复用评估结果，不是首次未触碰test。

逐样本预测：

```text
artifacts/experiments/phase8/official_test/P7-E01_seed42/test_predictions.csv
SHA-256:
5A4E3375E7A283280963A008BDD2AC3B670C830A88A837639DA026476FF5A713
```

### 8.3 P7-E04：导师规定核心方法

| 指标 | official test结果 |
|---|---:|
| ROC AUC | **0.6803666921** |
| PR AUC | 0.5865045233 |
| sensitivity | 0.7226890756 |
| specificity | 0.5272727273 |
| accuracy | 0.6091549296 |
| balanced accuracy | 0.6249809015 |
| precision | 0.5243902439 |
| F1 | 0.6077738516 |
| TN / FP / FN / TP | 87 / 78 / 33 / 86 |

使用阈值`0.7634837635542924`。

逐样本预测：

```text
artifacts/experiments/phase8/official_test/P7-E04_seed42/test_predictions.csv
SHA-256:
1A01673A033FFFA033F9494E215DE5A177179AC74F1BB4BCFEF0C265E557B601
```

### 8.4 validation至test趋势

| 模型 | seed 42 val ROC / PR | test ROC / PR | test−val ROC / PR |
|---|---:|---:|---:|
| P7-E01 V2S | 0.7179266896 / 0.6056954281 | 0.7219760632 / 0.6122469838 | +0.0040493736 / +0.0065515557 |
| P7-E04 V2S+BiFPN+CBAM | 0.6679553265 / 0.5554048313 | 0.6803666921 / 0.5865045233 | +0.0124113657 / +0.0310996920 |

P7-E01在validation和official test上的ROC AUC、PR AUC均高于P7-E04，因此validation的方向性趋势延续到了test。但该结果不改变两模型预先声明的角色，也不触发重新选模或调参。

### 8.5 患者级配对bootstrap

固定随机种子42，以patient_id为聚类单元，每次保留该患者全部视图，共生成2,000个有效重采样：

| P7-E01 − P7-E04 | 点差 | 95% CI |
|---|---:|---:|
| ROC AUC | +0.0416093710 | [-0.0253836888, 0.1105897027] |
| PR AUC | +0.0257424606 | [-0.0498616422, 0.1043560696] |

两个区间均跨0。因此可以报告P7-E01点估计更高且方向与validation一致，但不能据此宣称两模型差异具有统计确定性。

配对结果：

```text
artifacts/experiments/phase8/official_test/paired_comparison/paired_bootstrap.json
SHA-256:
09CDFF62AD9F05484AF68A01A88DC7344CF2F98CE9AF7FD47287C03485C24EDA
```

Gate B未发生失败或重跑。P7-E01和P7-E04各完成1个新pass，项目累计official test模型推理pass从1增至3。

## 九、Gate C：正式Grad-CAM++解释性比较

### 9.1 授权与ROI几何审计

Gate C授权记录允许读取official test真实图像和ROI mask，但不允许训练、阈值重选、额外分类test pass或Gate D：

```text
artifacts/experiments/phase8/gradcam_pp_authorization.json
SHA-256:
7B1A51E9C41A312886FACC03E60FE6EB35C07566FB1173C2AD7836CC580AED26
```

首次几何审计因白名单只接受`cropped images`、未接受唯一一条合法`ROI mask images`系列而在模型加载前停止。失败记录已保留，没有删除或覆盖。修订仅把IDC metadata系列类型白名单固定为这两个值，attempt-2使用新目录并通过：

| 几何审计项目 | 结果 |
|---|---:|
| official test图像 | 284 |
| official ROI记录 | 326 |
| 成功解析且坐标可比的ROI记录 | 324 |
| 坐标排除ROI记录 | 2 |
| ROI完整图像 | 282 |
| 恶性图像 | 119 |
| 可进入定量定位的恶性图像 | 117 |
| 坐标排除恶性图像 | 2 |

所有324条有效ROI均满足：

- official记录的patient、laterality和CC/MLO与test manifest一致；
- ROI DICOM为非空全尺寸二值mask；
- ROI与对应whole mammogram的Rows、Columns完全一致；
- ROI与whole mammogram的`PatientOrientation`均与official CC/MLO一致；
- 对同一whole mammogram的多条ROI先在原像素网格合并为union mask；
- image与union mask使用完全相同的breast crop和letterbox几何变换。

患者`P_00353`的LEFT CC与LEFT MLO两条异常mask尺寸分别与whole mammogram不一致。两张图保留Grad-CAM++定性结果，但标记为`not_comparable_roi_geometry`，不进行任意缩放，也不进入IoU/Dice等统计。

attempt-2权威结果：

```text
artifacts/experiments/phase8/gradcam_pp_geometry_audit_attempt2/outcome.json
SHA-256:
69446B3C9BF0020396D1E87CAA0F311868362568DE86B111D62A4323F0087BAD
```

12张固定样本的绿色ROI轮廓叠加图已人工查看，没有发现镜像、旋转或letterbox偏移。

### 9.2 正式运行范围

- 固定样本：全部119张恶性图像，加固定定性面板中的6张良性图像，共125张；
- 固定模型：P5-E02、P7-E01、P7-E04，各seed 42；
- 固定目标层：`channel_projection`、`channel_projection`、`token_compressor.mixer`；
- 固定目标：恶性单logit；
- 输入与训练一致：384、breast crop+letterbox、ImageNet归一化；
- float32反向、AMP关闭、batch size 2；
- 有效乳房区域内min-max归一化，padding不参与；
- 唯一二值阈值：热度top 10%；
- 患者级bootstrap：2,000次、seed 42、95% CI；
- 共生成375个image-model解释，189个反向batch；
- 没有新增分类official test pass，项目累计分类test pass仍为3。

正式执行前另一次预检因P5配置省略三个训练期增强字段、而P7显式记录为0而停止；逐字段审计证明所有推理相关配置相同，且省略值的运行默认值也是0。该停止发生在正式输出创建和模型加载前，记录已保留。之后正式模型运行只执行一次并成功完成。

### 9.3 图像级定位结果

以下均为图像级均值，区间为patient_id聚类bootstrap 95% CI：

| 模型 | 有效图像 | IoU | Dice | lesion hit rate | energy inside ROI |
|---|---:|---:|---:|---:|---:|
| P5-E02 B0 | 117 | 0.02029 [0.00796, 0.03449] | 0.03374 [0.01390, 0.05649] | 0.01709 [0.00000, 0.04311] | 0.04599 [0.02452, 0.07199] |
| P7-E01 V2S | 116 | **0.03223** [0.01489, 0.05218] | **0.05187** [0.02536, 0.08208] | 0.00862 [0.00000, 0.02679] | **0.05676** [0.03048, 0.08913] |
| P7-E04 V2S+BiFPN+CBAM | 117 | 0.01490 [0.00833, 0.02233] | 0.02667 [0.01555, 0.03930] | 0.00000 [0.00000, 0.00000] | 0.04511 [0.03113, 0.06201] |

P7-E01在样本`6ad81fbdd4b3c90e`上产生退化全零热力图。该图记录为`degenerate_heatmap`并从定位指标中排除，没有使用伪造的0分，因此P7-E01有效图像数为116。

### 9.4 配对解释性比较

在三个模型共同有效的116张恶性图像上：

| 配对差值 | IoU | Dice | lesion hit | ROI energy |
|---|---:|---:|---:|---:|
| P7-E01 − P5-E02 | +0.01176 [-0.00250, 0.02619] | +0.01784 [-0.00473, 0.04046] | -0.00862 [-0.04168, 0.01770] | +0.01038 [-0.00894, 0.02781] |
| P7-E01 − P7-E04 | +0.01720 [-0.00176, 0.03836] | +0.02498 [-0.00420, 0.05687] | +0.00862 [0.00000, 0.02679] | +0.01126 [-0.01001, 0.03414] |

P7-E01的IoU、Dice和ROI energy点估计最高，但对应配对区间均跨0；相对P7-E04的lesion-hit区间下界为0。因此不能声称P7-E01在解释性定位上具有统计确定的优势。

### 9.5 定性结论与产物

固定12张样本的对照面板显示，三模型的高响应常位于乳房边缘、乳头附近或大范围致密组织，而非集中覆盖钙化ROI。这与整体较低的IoU、Dice和lesion-hit结果一致。Grad-CAM++在本项目中提供的是模型注意区域证据，不证明病灶级定位能力、因果机制或临床可靠性。

```text
正式结果：
artifacts/experiments/phase8/gradcam_pp_official_test/outcome.json
SHA-256:
9E813F198E37C12271643B21B05A3B39A7C1E57272A837F9847816DDFB811F27

逐图指标：
artifacts/experiments/phase8/gradcam_pp_official_test/per_image_results.csv
SHA-256:
3E408D9B267ED07C24EF5091125D3EFC27BE01A3BC7E352DDA427B005C806DBB

配对比较：
artifacts/experiments/phase8/gradcam_pp_official_test/paired_localization_comparisons.json
SHA-256:
59C8427BA3250E80EB896169F63A189F36A984787021FBAF9760AC9E662CE951

固定面板：
artifacts/experiments/phase8/gradcam_pp_official_test/qualitative/qualitative_panel.png
SHA-256:
77F2D9BC1A8F62A5C0F981D8DC72689F3E402B751203F2AC74A2303F0D27B01C
```

三个模型的125张`384 × 384`有限值机器可读heatmap数组均已保存并与逐图sample_id顺序一致。

## 十、配置与入口哈希

```text
configs/phase8/artifact_lock.yaml
F360DBF32FF1035F9A4E7FE6BD1003387BAF1A79D80486FAE5A5A8BDDF980115

configs/phase8/official_test.yaml
B4503F5B46960C9386662D9763FE6039503ABDFCAD8B1B917BAE59A00DEB8738

configs/phase8/gradcam_pp.yaml
9DF9D89CE590796E6B85AE47943C4372E7CBA8FA5568057BE9F32ACAD116F5F0

scripts/run_phase8_official_test.py
9D383E4525A215C5194CDABA1E6427336B4F4B999052B584473094B401550147

scripts/audit_phase8_roi_geometry.py
AD8E698ABB9D669801A4D8292B37D5D9E8709228FAFACBF40DE989701E5A5CD7

scripts/run_phase8_gradcam_pp.py
F2EA374A8409B50BB54A29DDD7B7BB78A3F04E975BEC78CC1EBCD93CA47594F8

src/cbis_ddsm/phase8_xai.py
FDEE5FE230E4802A310410EB0BF61C67538BDE3A5B32853197DF4604B05FB81D
```

## 十一、测试结果

- Python语法编译：通过；
- Gate B实施前阶段八专项测试：`12 passed in 5.64s`；
- Gate B实施前完整pytest：`118 passed in 15.25s`；
- Gate B完成后完整pytest：`118 passed in 13.66s`；
- Gate C正式运行前专项测试：`11 passed in 6.50s`；
- Gate C正式运行前完整pytest：`123 passed in 12.30s`；
- Gate C完成后完整pytest：`123 passed in 13.31s`；
- Gate D初始数据审计前专项测试：`13 passed in 23.23s`；
- Gate D隐藏XLS工作表审计完成后专项测试：`14 passed in 4.21s`；
- Gate D初始数据审计后完整pytest：`136 passed in 13.58s`；
- Gate D attempt-2完成后完整pytest：`137 passed in 12.33s`；
- Gate D正式推理前专项测试：`16 passed in 5.38s`；
- Gate D正式推理前完整pytest：`139 passed in 13.69s`；
- Gate D完成后完整pytest：`139 passed in 13.42s`；
- `git diff --check`：通过；
- 阶段一至阶段七测试无回归。

Gate B完成后只对已保存预测执行离线复核，没有重新运行模型。复核确认每个预测文件均为284行、151名患者、284个唯一sample_id、同时包含两类，ROC AUC与PR AUC和正式结果完全一致。

## 十二、Gate D：INbreast数据审计与探索性迁移

### 12.1 授权与原始数据纪律

用户指定只读数据根目录：

```text
D:\School Work\codex项目\数据集\dataset\inbreast
```

Gate D授权记录：

```text
artifacts/experiments/phase8/external/inbreast_authorization.json
SHA-256:
F0D979BBA3F1C82F1F74B9EA3E0D5C94A0E7DF6DFB9890E015B1B3D5F3A4F86C
```

路线1授权记录：

```text
artifacts/experiments/phase8/external/inbreast_route1_authorization.json
SHA-256:
0DCAF42B12E19FF71C60E3ABB8FAA9DCD31F590C3C5B92E58D4BCA1FF3F56187
```

全过程没有移动、删除、重命名或覆盖原始文件，没有安装依赖，也没有重新读取CBIS-DDSM official test。

### 12.2 来源、文件与DICOM审计

INbreast Release 1.0压缩包SHA-256为：

```text
A536DA310985458906F645D8AA998451244F975385E85E941D30C9AFC6134DB6
```

数据清单与解码结果：

| 项目 | 结果 |
|---|---:|
| 清单文件 | 1,622 |
| 总字节数 | 11,080,401,277 |
| DICOM图像 | 410 |
| XML标注 | 343 |
| OsiriX ROI | 343 |
| 医疗报告 | 117 |
| CSV汇总行 | 410 |
| 文件名恢复患者 | 108 |
| DICOM全量解码 | 410/410通过 |
| 有限像素 | 410/410通过 |
| Photometric Interpretation | 全部`MONOCHROME2` |
| Transfer Syntax | 全部`1.2.840.10008.1.2` |
| CSV与文件名侧别/视图不一致 | 0 |

所有源文件SHA-256清单：

```text
artifacts/experiments/phase8/external/inbreast/audit_attempt2/source_file_sha256.csv
SHA-256:
4DDF10C3C62E834984876F68140EEC2665365EDE5BBDF4521EF996207B2488DC
```

### 12.3 钙化候选与标签口径

XML ROI名称审计把`Calcification`、`Calcifications`与`Cluster`固定为calcification-related标记：

| 项目 | 结果 |
|---|---:|
| calcification-related图像 | 315 |
| 排除不支持的FB视图 | 1 |
| 候选CC/MLO图像 | 314 |
| 候选患者 | 91 |
| BI-RADS 1/2/3候选图像 | 227 |
| BI-RADS 4a/4b/4c/5/6候选图像 | 87 |

初始attempt-1只依据README、CSV、XML和医疗报告完成审计，没有解析XLS隐藏工作表；该结果按不可覆盖纪律保留，但不作为最终标签结论。随后使用系统已有的只读Excel OLE DB驱动完成attempt-2：

- `Sheet1`包含410条可用图像记录，统一标签字段仍为`Bi-Rads`；
- 隐藏`Sheet2`存在`DG`列，但只有55行非空；
- `Sheet2`不含`File Name`、laterality或view，无法直接连接到DICOM图像；
- `Sheet1`与`Sheet2`同一行的BI-RADS有260处不一致，证明不能把两张表按行号拼接；
- 因此55条`DG`诊断不能形成确定的图像级病理标签。

INbreast README明确说明汇总表提供BI-RADS分类；完整XLS虽存在部分`DG`诊断，但缺少可验证的图像级连接键。医疗报告也不是统一标签表，不能据此自动为全部图像构造病理ground truth。因此：

- 当前数据不能支持与CBIS-DDSM病理`MALIGNANT/BENIGN`等价的正式外部分类声明；
- 用户已确认采用`1/2/3 → 0`、`4a/4b/4c/5/6 → 1`的探索性BI-RADS迁移；
- 另一条合法路线是只使用XML标注进行解释性定位验证；
- 本次审计发现67个XML点超出对应DICOM像素边界；若选择定位路线，必须先进行更细的坐标语义审计，不得直接计算IoU/Dice。

### 12.4 审计结论与产物

审计状态：

```text
AUDIT_PASS_ROUTE_CONFIRMATION_REQUIRED
```

正式结果：

```text
artifacts/experiments/phase8/external/inbreast/audit_attempt2/outcome.json
SHA-256:
CE390321B1262D0CE2424C5C75A344E277CD9A11BE130CBF2C35D0078F2EF900
```

候选manifest：

```text
artifacts/experiments/phase8/external/inbreast/audit_attempt2/candidate_manifest.csv
SHA-256:
E02A180291DF89971B34295D48284B60EF289880A71E0BCF9B0CDC8ACBB7D93A
```

用户已选择`exploratory_birads_transfer`；`annotation_localization_only`未执行。

### 12.5 路线1冻结外部推理

正式样本固定为314张calcification-related CC/MLO图像、91名患者，其中映射阴性227张、映射阳性87张。两个模型均使用：

- 原阶段七seed 42冻结checkpoint；
- 384输入、breast crop+letterbox和ImageNet归一化；
- 无训练、无微调、无域适配、无TTA、无ensemble；
- 各自在CBIS-DDSM validation冻结的Youden J阈值；
- 患者级2,000次bootstrap、seed 42、95% CI；
- `official_test_loaded=false`与`test_split_loaded=false`。

主要探索性排序指标：

| 模型 | ROC AUC [95% CI] | PR AUC [95% CI] |
|---|---:|---:|
| P7-E01 V2S | **0.56150** [0.47912, 0.64035] | **0.34800** [0.25409, 0.45238] |
| P7-E04 V2S+BiFPN+CBAM | 0.54775 [0.44286, 0.64393] | 0.32719 [0.23196, 0.45667] |

使用冻结阈值的探索性分类指标：

| 模型 | sensitivity | specificity | accuracy | balanced accuracy | precision | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| P7-E01 | 0.24138 | 0.82819 | 0.66561 | 0.53479 | 0.35000 | 0.28571 | 188 / 39 / 66 / 21 |
| P7-E04 | 0.25287 | 0.83700 | 0.67516 | 0.54494 | 0.37288 | 0.30137 | 190 / 37 / 65 / 22 |

患者级配对差值：

| P7-E01 − P7-E04 | 点差 | 95% CI |
|---|---:|---:|
| ROC AUC | +0.01375 | [-0.07090, 0.10442] |
| PR AUC | +0.02081 | [-0.06583, 0.09186] |

两项配对区间均跨0，不能宣称P7-E01在该探索性外部任务上具有统计确定优势。两个模型的ROC AUC均接近0.5，且敏感度约0.24–0.25；较高accuracy主要受227/87类别不平衡和高特异度影响，balanced accuracy更接近随机水平。这表明从CBIS-DDSM病理钙化分类向INbreast BI-RADS映射任务的跨域迁移能力较弱。

路线1正式结果：

```text
artifacts/experiments/phase8/external/inbreast/formal/outcome.json
SHA-256:
F186B5F5F1D2B7C2A41FEBC8FD5F4F3C520D30E969F430BB8AD6B1FF27DB4B81
```

逐图预测：

```text
P7-E01:
artifacts/experiments/phase8/external/inbreast/formal/P7-E01_seed42/external_predictions.csv
SHA-256:
12ABE1D290F654743A44996E3893339EC6834D8C68355ACB9B531FF1809DD738

P7-E04:
artifacts/experiments/phase8/external/inbreast/formal/P7-E04_seed42/external_predictions.csv
SHA-256:
35D0631D10AF83FF321F9D5B9989EC63C26887C8AB875546F3FF112CD7AE14FA
```

配置与入口：

```text
configs/phase8/inbreast_external.yaml
52CCD4FC625C891DD4A421F24BDA2A528E6882DC96256722B792BB3B2D3B62D6

src/cbis_ddsm/phase8_external.py
07769996B95810B068D44908FC21CE2305FFD9D19A1CA60EB081E9ECFB417269

scripts/run_phase8_external_validation.py
2809D8B3DF14DB47720C16E061D25660359D9F2B2CB0288227FE10B2341FD338
```

## 十三、Gate状态

| Gate | 状态 | 结论 |
|---|---|---|
| Gate 0 | 完成 | 分支、manifest、checkpoint、预测、哈希和严格加载通过 |
| Gate A | 完成 | 配置门、阈值冻结、Grad-CAM++基础实现完成 |
| Gate B | 完成 | 两个冻结模型各1个official test pass、离线指标与患者级配对比较完成 |
| Gate C | 完成 | ROI几何审计、三模型375组正式解释、定位指标、配对比较与固定面板完成 |
| Gate D | 完成 | INbreast审计及P7-E01/P7-E04探索性BI-RADS迁移完成；不作病理等价声明 |

阶段八Gate 0至Gate D均已完成。当前授权不包含依赖安装、commit或push。
