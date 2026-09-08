# Frozen GBSA-TCL Prospective32 FINAL

日期：2026-09-08  
评价口径：MatterGen 官方结构指标 + Alex-MP + MatterSim-5M surrogate；20,000 次 paired bootstrap。  
本报告不修改历史 GBSA-TCL P1 结论，也不启动 independent64/Formal256/Atomic-only。

1. **Branch**：`experiment/gbsa-tcl-prospective32`。

2. **Final HEAD**：本报告随最终结果提交归档；精确 SHA 在任务最终交付消息中给出。预注册提交为 `3540cb14bbf3cebaf215d7bdcc1af17e18a491df`，实验起点为 `b12985130a7e7dfff490118d566fefd1e2b65af2`。

3. **Seed range**：`87000–87031`，32 个 paired seeds；C0/GBSA-TCL 共 64 次真实生成。

4. **Historical overlap**：当前可读仓库树及已知历史区间未发现重叠。完整 `git log --all -G` 扫描受到仓库既存损坏对象 `f44a68…` 无法解包的限制，因此这里只声明“可读历史无重叠”，不作无法验证的绝对审计承诺。

5. **Training steps**：0；没有重新训练或微调。

6. **GBSA checkpoint SHA256**：`1edc1c0c7848e9edaff9d73b135642fcfaa36e23591e4e2f336c39b3df3dcf0f`。C0 base checkpoint SHA256：`01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`。

7. **C0 generation success**：32/32（100%）；平均每样本生成时间 117.723 s。

8. **GBSA generation success**：32/32（100%）；平均每样本生成时间 116.086 s。

9. **Technical reruns**：0。

10. **MatterSim success**：C0 32/32、GBSA-TCL 32/32；均为真实 CUDA MatterSim-5M 逐结构松弛，任务失败数 0。

11. **DFT_VERIFIED**：`False`。E-hull、force 与 relaxation 都是 MatterSim-5M surrogate 结果，不代表 DFT 或实验验证。

## PRIMARY

12. **C0 E-hull**：mean 0.078758、median 0.071429、P95 0.169020、max 0.231144 eV/atom。

13. **GBSA E-hull**：mean 0.108419、median 0.054776、P95 0.325742、max 1.194370 eV/atom。

14. **ΔE-hull**：mean(GBSA−C0) = +0.029661 eV/atom；paired W/T/L（越低越好）= 18/0/14。

15. **E-hull 95% CI**：[-0.033450, +0.118224] eV/atom。

16. **E-hull NI margin**：+0.010 eV/atom。

17. **E-hull verdict**：**BORDERLINE**。CI upper 超过 +0.010，不能证明非劣；CI lower 又未高于 +0.010，不能判明确 FAIL。

18. **C0 NUS**：12/32 = 37.50%；Clopper-Pearson 95% CI 21.10%–56.31%。

19. **GBSA NUS**：18/32 = 56.25%；Clopper-Pearson 95% CI 37.66%–73.64%。

20. **ΔNUS**：+18.75 percentage points。

21. **ΔNUS paired bootstrap 95% CI**：[-6.25, +43.75] pp。

22. **NUS paired transitions**：C0 fail→GBSA success 12；C0 success→GBSA fail 6；两者均 success 6；两者均 fail 8；W/T/L（越高越好）= 12/14/6；McNemar exact p=0.237885。改善 seeds：87001、87003、87005、87007、87008、87012、87013、87014、87015、87023、87028、87031；退化 seeds：87000、87010、87011、87017、87022、87025。

23. **NUS verdict**：**FAIL**。点估计满足 ≥+5 pp，但预注册要求 CI lower >0；实际 lower=-6.25 pp。

## SECONDARY

24. **Stable**：C0 19/32=59.375%，GBSA 24/32=75.000%，Δ=+15.625 pp，paired CI [-9.375,+40.625] pp；fail→success 12、success→fail 7、ties 13。

25. **Novel**：C0 21/32=65.625%，GBSA 26/32=81.250%，Δ=+15.625 pp，paired CI [-3.125,+34.375] pp；fail→success 7、success→fail 2、ties 23。

26. **Unique**：C0 32/32=100%，GBSA 32/32=100%，Δ=0 pp，paired CI [0,0]；无 exact duplicate/mode-collapse 信号。

27. **Secondary interpretation**：NUS 点估计提升同时伴随 Stable 与 Novel 点估计各 +15.625 pp，而非以牺牲 novelty/unique 换取；但两者 CI 均跨 0。额外的官方 MatterGen validity 显示 structure validity 两组均 32/32；SMACT composition validity 从 C0 30/32=93.75% 降至 GBSA 26/32=81.25%，Δ=-12.50 pp、paired CI [-25.00,-3.125] pp，是不利信号，但不是本轮预注册 primary/direct-fail 条款。

## PHYSICAL GUARDRAILS

28. **RMSD**：C0 mean/median/P95/max = 0.025717/0.012712/0.084592/0.154341 Å，RMSD>0.5 为 0；GBSA = 0.107265/0.023738/0.537371/1.393174 Å，RMSD>0.5 为 3。Δmean=+0.081548 Å，paired CI [+0.007344,+0.184232] Å，margin=+0.020 Å，**BORDERLINE**（upper 超 margin，lower 未超过 margin）。

29. **Atomic force（初始结构，逐结构原子力范数均值）**：C0 mean/median/P95/max = 0.117592/0.075909/0.351476/0.624329 eV/Å；GBSA = 0.126580/0.056023/0.427070/0.899676 eV/Å。绝对 mean delta=+0.008987 eV/Å；mean ratio=1.076429，paired bootstrap ratio CI [0.584762,1.886600]，margin=1.20，**BORDERLINE**。

30. **MaxF distribution（初始结构）**：C0 median/P95/P99/max = 0.146090/0.921443/1.070078/1.131341 eV/Å；GBSA = 0.107916/0.967972/1.688151/1.974433 eV/Å。单一 maximum 不直接决定 verdict。

31. **MaxF>1 eV/Å**：C0 1/32=3.125%（exact CI 0.079%–16.217%）；GBSA 2/32=6.25%（0.766%–20.807%）。paired delta=+3.125 pp，bootstrap CI [-6.25,+15.625] pp；C0 no-event→GBSA event 2、反向 1；margin=+5 pp，**BORDERLINE**。

32. **MaxF>2 eV/Å**：C0 0/32=0%、GBSA 0/32=0%；各自 exact CI 0%–10.888%；paired delta=0 pp，bootstrap CI [0,0] pp；margin=+2 pp，**PASS**。

## DIAGNOSTICS

33. **Relaxation**：C0 steps mean/median/P95/max = 32.563/25.5/68.85/167，>200=0、>400=0；GBSA = 65.625/31/250.10/658，>200=2、>400=2。GBSA 有明显更重的松弛长尾，但 relaxation 仅作 diagnostic。既有 late-position score-drift 机制分析不重复；因会产生额外计算，本轮按预注册要求 skip。

34. **Invalid structures**：technical invalid=0/64；生成态、初始带力结构和松弛态结构均满足 finite coordinates/cells/forces 与 positive cell determinant；finite energies 64/64。MatterGen structure validity 两组均 32/32。SMACT composition-invalid 为 C0 2/32、GBSA 6/32，已完整计入而未删除。

35. **Non-converged structures**：松弛任务失败 0，但 frozen runner 不提供显式 optimizer-converged 字段，不能把 job success 等价为严格收敛。最终 MaxF>0.05 eV/Å 的诊断计数为 C0 0、GBSA 2（87010=0.052224，87027=0.050113 eV/Å）；这两项保留为诊断，不冒充 ExpCellFilter convergence 判据。

36. **Force leave-one-out sensitivity**：主分析保留全部 32 seeds。完整 atomic-force ratio=1.076429；逐一删除一个 seed 后 ratio 范围 0.867694–1.272649，说明单样本可改变 ratio 相对 1 的方向。32 个 LOO 的 atomic-force verdict 全为 BORDERLINE；MaxF>1 全为 BORDERLINE；MaxF>2 全为 PASS。删除 87013 时 ratio 最低 0.867694；删除 87001 时最高 1.272649。LOO 不改变主 verdict。

37. **所有 severe GBSA seeds**（均保留）：

   - 87005 `Eu2Au3`：E-hull 0.021664，RMSD 1.393174 Å，AtomicF 0.083168，MaxF 0.173792 eV/Å，425 steps；flags=RMSD>0.5、Steps>200、Steps>400。
   - 87010 `LiGd3Fe4SO7`：E-hull 0.373387，RMSD 0.507142 Å，AtomicF 0.386526，MaxF 1.050943 eV/Å，658 steps，final MaxF 0.052224；flags=RMSD>0.5、MaxF>1、Steps>200、Steps>400、FinalMaxF>0.05。
   - 87013 `GdPt3`：E-hull 0.022396，RMSD 0.089603 Å，AtomicF 0.899676，MaxF 1.974433 eV/Å，53 steps；flag=MaxF>1。
   - 87022 `GdPt3`：E-hull 1.194370，RMSD 0.040583 Å，AtomicF 0.390117，MaxF 0.517456 eV/Å，34 steps；flag=E-hull>0.5。
   - 87027 `Gd2Pt3`：E-hull 0.018474，RMSD 0.102112 Å，AtomicF 0.134676，MaxF 0.190361 eV/Å，107 steps，final MaxF 0.050113；flag=FinalMaxF>0.05。
   - 87030 `HgPt3`：E-hull 0.221529，RMSD 0.574318 Å，AtomicF 0.054027，MaxF 0.095003 eV/Å，59 steps；flag=RMSD>0.5。

## FINAL

38. **Primary NUS**：**FAIL**。

39. **Primary E-hull**：**BORDERLINE**。

40. **RMSD guardrail**：**BORDERLINE**。

41. **Atomic force guardrail**：**BORDERLINE**。

42. **Max-force tail**：**BORDERLINE**（>1 BORDERLINE；>2 PASS；没有明确 tail FAIL）。

43. **Novel/Unique collapse**：**NO**。

44. **Historical GBSA P1 verdict**：**FAIL（保持，不被本轮推翻）**。

45. **Prospective32 verdict**：**FAIL**，依据新的预注册 Primary + Guardrail framework。

46. **是否值得进入 frozen independent 64-seed validation**：**NO**；不得自动启动。

47. **具体失败/未通过处**：直接失败来自 NUS paired CI lower=-6.25 pp≤0。E-hull 非劣、RMSD、Atomic force 与 MaxF>1 均因区间过宽只到 BORDERLINE；另有 composition validity 下降及 GBSA relaxation/RMSD 长尾风险。

48. **是否建议停止 TCL/GBSA 路线并回退 E3-PCR**：**YES**。当前数据不足以支持继续扩大 Frozen GBSA；按既定 fallback 优先回到 E3-PCR 方向，但本轮不启动新实验。

49. **最终一句话科学结论**：在 checkpoint、评价标准与采样全部前瞻性冻结后，Frozen GBSA-TCL 在新 32 paired seeds 上取得了 +18.75 pp 的 NUS 点估计，但该提升缺乏 CI>0 的配对统计支持，且 E-hull 与三项物理安全性未证明非劣，因此结论为 FAIL、停止并不进入 independent64。
