# 摘要与主要贡献的修订建议（中英文对齐，未修改正文）

依据现有中英文摘要及第1章“本文主要创新点”，提出可直接用于下一轮修稿的**建议文本**。当前 PDF 与 LaTeX 仍是旧表述；本文件不声称已完成正文替换。新结果应在第3、5、6章同步加入后才可更新摘要，否则会出现“摘要有对照、正文无证据”的断裂。

## 旧表述 → 新表述原则

| 当前容易造成的印象 | 建议表述 | 原因 |
|---|---|---|
| “提出参考轨迹保留的多分支 CFG 引导方法” | “提出面向冻结扩散生成模型的**预算约束验证选择推理方法**；Fixed-K2 是其固定双候选实例” | 从具体 CFG 脉冲转到状态、预算、候选、验证、回退的接口；不冒充新训练模型。 |
| “Fixed-K2 提高生成质量” | “在 C1-128 上降低 **CHGNet 代理 Property MAE**，同时通过预定义的代理质量护栏，代价约为 C0 的 2.2× MatterGen score 调用” | “生成质量”笼统，容易被误读为 DFT 稳定性或所有指标同时改善。 |
| “K=2 是最佳搜索宽度/优于 Best-of-N” | “K=2 是既有固定预算实例；新宽度实验显示预算—误差权衡，**未证明 K2 最优**；独立 Best-of-2 数值更低且预算更少，但直接差异 CI 跨零” | 客观呈现不利对照，不把点估计写成统计定论。 |
| “额外计算显示轨迹分支的独立价值” | “额外推理计算配合冻结验证选择本身能改善代理误差；本文研究**如何在参考保持与预算约束下可靠选择**，尚未证明共享前缀优于独立生成” | Best-of-2 是强竞争基线；2.0×/2终点与2.2×/3终点不等预算。 |
| “学习型分配器进一步优化” | “Linear-K2 在 Phase B 和 C1 未获得独立支持” | 保留冻结负结论，不把探索写成贡献。 |
| “自适应预算方法” | “自适应预算是未来工作，需单独校准和新种子确认” | 当前只设计，未实现或验证。 |

## 建议替换：中文摘要中创新点1段落

> 针对冻结晶体扩散模型中固定单次采样的条件属性误差与在线引导偏离风险，本文提出面向冻结扩散生成模型的预算约束验证选择推理方法。该方法保留可精确复现的 C0 参考轨迹及随机状态，在预先限定的推理预算内展开候选后缀，并依据冻结的代理属性误差、结构有效性和质量条件进行终点选择；不满足接受条件时返回参考输出。Fixed-K2 是该框架的固定双候选实例。在独立 C1-128 配对实验中，它将 CHGNet 代理 Property MAE 从 0.034796 降至 0.026332（相对降低 24.32%），满足预定义的代理质量护栏，MatterGen score 调用约为 C0 的 2.2 倍。另一个全新 128 对样本中的独立 Best-of-2 在 2.0 倍预算下取得数值上更低的代理误差，二者直接配对差异的 95% 置信区间跨零。因此，本文不声称 Fixed-K2 优于独立多样本筛选；学习型 Linear-K2 的额外优势亦未得到独立支持。

若学校摘要字数严格受限，保留“独立 Best-of-2 数值更低、未证实优于它”这一完整边界，可以把 95% CI 的具体数值移至正文，不能仅删除不利结果。Budget Scaling 可在摘要结尾增加一句，而不塞入全部 K 点：

> 在另一组独立的 K=0–4 预算实验中，选中结果的代理误差随候选预算增加而下降，但收益不均匀，既有 K=2 仅构成成本折中点。

注意：这一“下降”受嵌套候选与 SAFE-A 的选择规则影响，摘要不写成普适的推理扩展定律，更不写“真实材料质量随预算单调改善”。

## Suggested replacement: English abstract paragraph for Innovation 1

> To address conditional-property error and the risk of irreversible online guidance changes in a frozen crystal diffusion generator, this thesis proposes a **Budget-Constrained Verifier-Guided Inference Framework**. It preserves an exactly reproducible C0 reference trajectory and its random state, expands predeclared candidate suffixes within an explicit inference budget, and selects a terminal structure only if frozen surrogate-property and quality constraints are satisfied; otherwise, it returns the reference. Fixed-K2 is a fixed-width instance of this framework. On the independent paired C1-128 cohort, it reduced CHGNet-surrogate Property MAE from 0.034796 to 0.026332 (24.32% relative reduction) while satisfying the prespecified surrogate guardrails, at approximately 2.2 times the MatterGen score-call budget of C0. In a separate fresh 128-seed comparison, independent Best-of-2 achieved a numerically lower surrogate error at a 2.0-times score-call budget, and the paired confidence interval for its direct contrast with Fixed-K2 included zero. We therefore do not claim superiority over independent multi-sample selection; a learned Linear-K2 allocator also showed no independently confirmed additional benefit.

英文摘要如果加入 K0–4，也应同步保留“nested selector-induced nonincrease”性质，不可把 `compute` 简化成 wall-clock speedup。

## 建议替换：第1章“主要创新点1”两段

> **面向冻结扩散生成模型的预算约束验证选择推理方法。** 本文把推理时的候选生成与终点筛选写成显式的状态—动作—转移—验证—预算接口：状态保留晶体扩散变量、步序、条件和 RNG；动作是预先冻结的候选后缀；MatterGen 采样器完成转移；CHGNet 属性误差及 MatterSim/几何质量代理构成终点约束；无合格候选时精确返回 C0。Fixed-K2 在第400步共享前缀并展开 GPulse、PPulse 两个后缀，是固定预算的具体实例，而非新的最优 CFG 调度器或学习型规划器。
>
> 在 C1-128 上，Fixed-K2 相对 C0 将代理 Property MAE 从 0.034796 降至 0.026332（24.32%），配对增益的 95% 置信区间为 `[0.005817,0.011391]`，并通过预注册的代理质量护栏；其生成器调用预算为 2.2×。新增独立 Best-of-2 对照在另一个 128 对 cohort 中于 2.0×预算取得数值上更低的 MAE，直接差异区间跨零，因此当前证据**不支持** Fixed-K2 优于独立 Best-of-N。K0–4 结果刻画了冻结候选顺序下非均匀的成本—误差权衡，K2 不是最优性能点。Linear-K2 的学习型分配仍为负结果。贡献应限定为预算化、参考保持、代理约束选择与可审计回退的实现及评估。

第1章技术路线、章3章名、第5章结论、第6章总结需要同名同步；不要只修改摘要/创新点列表。第2项创新 RC-NFGD 的摘要和贡献不因重新定位创新点1而改变效应量：Formal256 的代理力改善、Property MAE 约 1.14% 恶化、POST 更强、无 DFT 与未证实协同均须继续保留。

## 摘要与贡献的出稿检查

1. 中英文术语一致：中文“预算约束验证选择推理方法”，英文 `Budget-Constrained Verifier-Guided Inference Framework`；`Fixed-K2` 始终指实例而非整个框架。
2. 24.32% 与 `[0.005817,0.011391]` 只对应历史 **C1-128**；Best-of-2 的 `0.037623/0.028097/0.029506` 只对应**另一新 cohort**，不得拼成一张“同样本”效果表。
3. Best-of-2 数值更低 ≠ 已被直接配对 CI 确认更优；Fixed-K2 的直接优势更未得到支持。
4. `Property MAE` 写“CHGNet 代理”；E-hull/Stable 写“MatterSim 等代理”；“安全”写“对冻结代理约束的参考回退”，不写物理安全保证。
5. 保留 Linear-K2、POST 和未确认联合协同的负结论；不加入 Adaptive Budget 已实现/已验证表述。
