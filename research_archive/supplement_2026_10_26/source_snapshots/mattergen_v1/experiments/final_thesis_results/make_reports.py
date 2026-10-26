"""Chinese thesis materials generated from audited full-precision result exports."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
F0=PROJECT.with_name('mattergen_v1_matersim_guidance')
MINI=PROJECT/'experiments/final_innovation1_audit/minimal_ablation'
TITLE='《基于深度学习的材料逆向生成》'
I1='多字段残差驱动的在线自适应 CFG'
I2='基于阶段可靠性校准的后期神经力场引导扩散方法'
I2EN='Reliability-Calibrated Late-stage Neural Force-Guided Diffusion'

def write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text+'\n',encoding='utf-8')
def table(d):return d.fillna('NA').to_markdown(index=False,floatfmt='.10g')+'\n'
def fmt(x):return f'{float(x):.10g}'
def pct(x):return f'{100*float(x):.4f}%'
def js(p):return json.loads(Path(p).read_text())

class Reports:
    def __init__(self):
        self.main=pd.read_csv(ROOT/'main_results.csv',float_precision='round_trip')
        self.stats=pd.read_csv(ROOT/'paired_statistics.csv',float_precision='round_trip')
        self.dist=pd.read_csv(ROOT/'distribution_statistics.csv',float_precision='round_trip')
        self.raw=pd.read_csv(ROOT/'all_per_seed_metrics.csv',float_precision='round_trip')
        self.old=js(ROOT/'original_decisions.json');self.audit=js(ROOT/'source_audits.json')
    def s(self,cohort,base,method,metric):
        d=self.stats;return d[(d.cohort==cohort)&(d.baseline==base)&(d.method==method)&(d.metric==metric)].iloc[0]
    def m(self,cohort,method,metric):
        d=self.main;return float(d[(d.Cohort==cohort)&(d.Method==method)].iloc[0][metric])
    def mt(self,cohort):
        d=self.main[self.main.Cohort==cohort]
        return table(d.drop(columns=['Runtime scope','Target dft_mag_density']))
    def st(self,cohort,metrics):
        d=self.stats;return table(d[(d.cohort==cohort)&(d.metric.isin(metrics))][['baseline','method','metric','n',
           'baseline_mean','method_mean','delta_method_minus_baseline','paired_median_delta','delta_ci95_low','delta_ci95_high','wins','ties','losses']])
    def effect(self,cohort,base,method,metric):
        r=self.s(cohort,base,method,metric)
        return (f'{metric}：{fmt(r.baseline_mean)} → {fmt(r.method_mean)}；相对改善 {pct(r.relative_improvement)}，'
                f'配对 95% CI [{pct(r.relative_ci95_low)}, {pct(r.relative_ci95_high)}]；'
                f'改善/持平/变差 {int(r.wins)}/{int(r.ties)}/{int(r.losses)}。')

    def contents(self):
        c=self.audit['Innovation1']['frozen_config'];f=self.audit['A1']['config']['F0']
        i1source=next(x for x in js(ROOT/'reproducibility_manifest.json')['git_sources'] if x['path'].endswith('innovation1/formal_final_report.json'))['exported_to']
        self.provenance=f'''本文所有数字由 `collect_results.py` 自动读取并按 seed 合并，不从提示词近似数抄写。完整精度见 [主结果 CSV]({ROOT}/main_results.csv)、[逐种子数据]({ROOT}/all_per_seed_metrics.csv)、[统计 CSV]({ROOT}/paired_statistics.csv)。

历史判定、阈值和置信区间保存在 [原始判定归档]({ROOT}/original_decisions.json)；原始来源、Git commit、SHA256、评价器和配置见 [复现清单]({ROOT}/reproducibility_manifest.json)。创新点一原始统计来源：[Formal256 报告]({i1source})。

统一边界：SURROGATE_PROPERTY_EVAL=True；DFT_VERIFIED=False。Stable/E-hull/力均为代理评价，不是实验合成或 DFT 真值证明。不同 cohort 不作绝对指标直接排名。'''
        self.i1method=rf'''在同一次条件/无条件得分计算后，对字段 $f\in\{{a,p,L\}}$ 计算

$$d_{{f,t}}=\sqrt{{\operatorname{{mean}}((s_{{c,f,t}}-s_{{u,f,t}})^2)}} ,\qquad d_t=\tfrac13\sum_f d_{{f,t}}.$$

这不是将不同形状张量拼接后的总体 RMS，而是每个字段单独降维后等权取均值。源码顺序为“先字段聚合、后阶段 EMA”。对 $q\in\{{predictor,corrector\}}$ 分别维护

$$m_{{q,t}}=\begin{{cases}}d_t,&\text{{该阶段首次观测}}\\\beta m_{{q,prev}}+(1-\beta)d_t,&\text{{否则}}\end{{cases}}.$$

先用当前观测更新 EMA，再求比值 $r_t=d_t/(m_{{q,t}}+\epsilon)$，并计算

$$u_t=\operatorname{{clip}}(1+\alpha(r_t-1),0.25,4),\quad g_t=\operatorname{{clip}}(g_0u_t,g_{{min}},g_{{max}}),$$

$$s_{{f,t}}=\operatorname{{lerp}}(s_{{u,f,t}},s_{{c,f,t}},g_t),\quad \forall f.$$

所有字段共用同一个 $g_t$。不是 field-specific CFG，不是 predictor/corrector 使用同一 EMA。冻结参数：$g_0={c['base_guidance']}$，$\alpha={c['adaptive_alpha']}$，$\beta={c['adaptive_ema']}$，$\epsilon={c['adaptive_epsilon']}$，最终 scale 区间 [{c['guidance_min_scale']}, {c['guidance_max_scale']}]。本方法使用 adaptive 而非 stage_adaptive，不叠加 piecewise 日程。非法残差/非有限统计回退至基础引导；每次采样开始重置 EMA。

源码：[guidance_schedule.py]({PROJECT}/mattergen/diffusion/sampling/guidance_schedule.py)、[classifier_free_guidance.py]({PROJECT}/mattergen/diffusion/sampling/classifier_free_guidance.py)。'''
        orig=self.old['Innovation1']['innovation1']['paired_statistics']
        rows=[]
        for name,r in zip(['E-hull (eV/atom)','Stable (fraction)','NUS (fraction)'],orig):
            rows.append({'Metric':name,'A0-C0':r['paired_mean_difference_right_minus_left'],
              'Original 95% CI low':r['bootstrap_95_ci_low'],'Original 95% CI high':r['bootstrap_95_ci_high'],
              'Raw p':r['raw_p_value'],'Holm p':r['holm_corrected_p_value'],
              'Wins/Ties/Losses':f"{r['right_method_wins']}/{r['ties']}/{r['right_method_losses']}"})
        self.i1results=self.mt('Innovation1 Formal256')+'\n原始正式统计（保留原检验，不替换为新 bootstrap）：\n\n'+table(pd.DataFrame(rows))+f'''
冻结门槛要求 E-hull 恶化不超过 0.02 eV/atom，Stable 与 NUS 下降不超过 3 pp，并且至少两个主指标方向改善。原始门槛通过，FORMAL_INNOVATION1_CONFIRMED=True。

必须区别“预设方向性门槛确认”和“统计显著优越”：三个主指标原始 CI 均跨零，不能宣称显著改善，也未执行严格的统计非劣效检验。RMSD 从 {fmt(self.m('Innovation1 Formal256','C0','RMSD'))} 增至 {fmt(self.m('Innovation1 Formal256','A0','RMSD'))} Å，应如实报告质量维度间并非一致受益。

该 Formal256 未保存属性预测与 pre-relax 力原始数据，Property MAE、Hit、pre-relax MaxF/mean force 标为 NA。归档的 maximum_force_ev_ang 是松弛后力，绝不能填入 pre-relax MaxF 列。新的小样本或 A5 不能回填历史 Formal256。'''
        self.i2method=rf'''MatterSim 是外部预训练神经势，不是本论文训练或提出的模型。RC-NFGD 的贡献对象是“如何将冻结神经力信号接入周期晶体反向扩散”，而非 MatterSim 本身。

仅在 predictor 的 $t\le {f['guidance_t_max']}$ 触发；冻结 1000 步采样实际触发 20 次，模型时间从 0.020 到 0.001。corrector、原子类别及晶胞 score 不加入外力引导。原始固定 CFG=2，Adaptive CFG 关闭。

由当前引导 score 构造完整 clean estimate：原子类别取 logits 的 argmax 并加类别 offset；分数坐标为 $\hat p_0=wrap(p_t+\sigma_p^2s_p)$；晶胞按源码恢复

$$\hat L_0=\big[L_t+\sigma_L^2s_L-(1-\alpha_L)L_{{limit}}\big]/\alpha_L.$$

在 $\hat x_0=(\hat a_0,\hat p_0,\hat L_0)$ 上由冻结 MatterSim 求笛卡尔原子力。将力去平移均值 $\tilde F_i=F_i-\operatorname{{mean}}_jF_j$，得到

$$\Delta r_i=\frac{{\eta\tilde F_i}}{{\max(\max_j\|\tilde F_j\|,F_{{ref}})}}.$$

冻结 $\eta={f['nominal_cartesian_step_a']}$ Å、$F_{{ref}}={f['force_reference_ev_a']}$ eV/Å；再执行硬上限 {f['hard_cartesian_cap_a']} Å。上述归一化实际上已限制最大步长不超过 {f['nominal_cartesian_step_a']} Å，因此更大的硬上限在当前配置下通常冗余；不是每个原子都固定移动同样长度。

使用行向量晶胞约定 $r=pL$，因此

$$\Delta p=\Delta r\hat L_0^{{-1}},\qquad\Delta s_p=\Delta p/c_{{score}}.$$

$c_{{score}}$ 必须是当前位置 ancestral predictor 的实际 `_get_coeffs` 系数。最终仅将 $s_p$ 替换为 $s_p+\Delta s_p$，继续执行反向扩散。该等式保证 predictor 中新增的分数坐标贡献为 $\Delta p$；其笛卡尔上限是在 clean-cell 坐标系下定义，不是整个采样轨迹净移动距离上限。

候选 clean 结构经 modulo 周期包裹后若最小原子间距不足 {f['min_periodic_distance_floor_a']} Å，或出现非有限输出、非正晶胞/非法 score 系数，则恢复未加物理修正的原始 score。没有额外候选 MaxF 下降判定，没有自适应 trust-region、闭环控制器或预算分配。

源码：[冻结 late_force_sampler.py]({F0}/experiments/mattersim_late_force_guidance_p0/late_force_sampler.py)。'''
        dg=self.old['diagnostic']
        self.reliability=f'''阶段选择来自独立于正式测试 cohort 的历史 clean-estimate 诊断，而不是在 Formal256 上挑选最好时间。诊断 n={dg['diagnostic_n']}，选定 t={dg['best_late_t_norm']} 的 clean MaxF 与最终 MaxF 的 Spearman 相关为 {fmt(dg['best_maxF_final_spearman'])}，有限/有效率 {pct(dg['valid_fraction_at_best_t'])}。能量与最终 E-hull 的相关仅 {fmt(dg['energy_per_atom_final_Ehull_spearman_at_best_t'])}，不支持把能量预测强行纳入主要机制。

这里的 reliability calibration 是离线诊断支持的阶段选择及力参考标定，不是额外学习出的可靠性模块。clean-x0 的设计动机是让势函数看到更接近晶体形态的估计；现有实验没有等预算 noisy-x_t 对照，因此不能声称已证明所有时间点或所有原子势下 clean-x0 都优于 noisy-x_t。诊断中 t=0 标签指最后 epsilon predictor 后的结构，不是数学 t=0 的模型调用。'''
        self.formal=self.effect('A1','C0','F0','MaxF')+'\n\n'+self.effect('A1','C0','F0','Atomic Force')+'\n\n'+self.effect('A1','C0','F0','RMSD')+'\n\n'+self.mt('A1')+'''
所有正式种子保留，未因高 MaxF 或不利结果删除样本。冻结质量护栏通过，但不等于每个质量指标都改善：磁密度 MAE 与 E-hull 的均值有小幅增加，NUS 小幅增加，Stable/Hit/几何 Validity 未变。主结论限定为本 checkpoint、目标与采样设置下的代理预松弛力改善。'''
        self.independent=self.effect('A1','C0','F0','Independent MaxF')+'\n\n'+self.effect('A1','C0','F0','Independent mean force')+'\n\n'+table(self.dist[(self.dist.Cohort=='A1')&self.dist.Metric.isin(['Independent MaxF','Independent mean force'])])+'''
CROSS_MLIP_GENERALIZATION=SUPPORTED。C0/F0 共 512 个结构已完整评估，冻结 CHGNet 0.3.0 原始原子力向量复算两种力指标通过。该证据反驳“改善仅体现于同一个 MatterSim 打分器”的最强说法，但不能排除训练数据重叠，也不等价于真实能量面保证。CHGNet 同时用于磁性护栏，因此独立性是“未参与力引导”的评价器独立性，不是完全未被项目使用过的盲测模型。'''
        tail=self.dist[(self.dist.Cohort=='A1')&(self.dist.Metric=='Independent MaxF')].set_index('Method')
        meanforce=self.dist[(self.dist.Cohort=='A1')&(self.dist.Metric=='Independent mean force')].set_index('Method')
        self.independent+=f'''\n\n跨势改善并不覆盖所有分位数：CHGNet MaxF P90 从 {fmt(tail.loc['C0','p90'])} 增至 {fmt(tail.loc['F0','p90'])}，P95 从 {fmt(tail.loc['C0','p95'])} 增至 {fmt(tail.loc['F0','p95'])} eV/Å；原子平均力的结构间中位数从 {fmt(meanforce.loc['C0','median'])} 增至 {fmt(meanforce.loc['F0','median'])} eV/Å。支持的是预先指定的平均力相关终点，而不是全分布或每个尾部指标都一致改善。'''
        self.direction=self.mt('A3')+'\n'+self.st('A3',['MaxF','Independent MaxF'])+'''
FORCE_DIRECTION_EFFECT=SUPPORTED，含义限定为预设方向对照下的描述性排序和配对统计支持。G1 的两种势 MaxF 均值均低于 G0/G2/G3/G4。G2=random，G3=shuffled，G4=anti-force；所有有修正组时刻、20 个事件、接受次数、最大范数匹配。G1 参考向量和原位 per-atom norm 断言用于约束扰动幅度；不是任意给随机组更大噪声。

G2/G3/G4 重放配对 G1 轨迹的范数信息；G4 是 G1 参考方向的取反，不是其自身当前轨迹的实时反梯度。随机构造包含去平移/保范数约束，不应写成无约束各向同性独立扰动。全部 pairwise CI 为补充未多重校正统计，不将每一条比较都包装成家族显著性证明。'''
        self.bound=self.mt('A4')+'\n'+self.st('A4',['MaxF','Independent MaxF'])+'''
BOUNDING_NOT_SUPPORTED_AS_CORE_MECHANISM。T0=Base，T1=冻结 F0，T2=去除范数相关饱和及硬上限，仍保留相同方向、阶段、score 映射和几何安全检查。T2 的均值更低，当前结果不支持“bounded 优于 unbounded”或“bound 是收益的必要原因”。

Bounded correction 应定位为机械可验证的安全/幅度约束设计，而不是实验已证明的性能核心或稳定性提升。T2 同时去掉归一化饱和与硬 cap，不是只删去本已冗余的 0.01 Å cap；该对照不能进一步分离两者。保留负结果，不调整 cap，不重新调参。'''
        post=self.s('A6','F0','POST','Independent MaxF')
        self.online=self.mt('A6')+'\n'+self.st('A6',['MaxF','Atomic Force','Independent MaxF','Independent mean force'])+f'''
ONLINE_VS_POST=MIXED（总体证据；主指标 MatterSim MaxF 的明确胜者是 POST）。

1. POST 是否显著降低 MatterSim MaxF？是。{self.effect('A6','C0','POST','MaxF')}
2. F0 是否优于等预算 POST？在主指标上否。{self.effect('A6','F0','POST','MaxF')} 不能据此声称在线接入优于直接后处理。
3. 独立 CHGNet 上谁更好？F0 的点估计稍低，但 POST-F0 的绝对差 CI 为 [{fmt(post.delta_ci95_low)}, {fmt(post.delta_ci95_high)}] eV/Å，跨零，无法确定谁更优。一个“对 C0 显著”而另一个“不显著”，并不构成二者显著差异。
4. POST 是否改变属性/NUS/Validity？POST 磁密度 MAE={fmt(self.m('A6','POST','Property MAE'))}，C0={fmt(self.m('A6','C0','Property MAE'))}；并非严格相同，但冻结护栏通过。NUS={pct(self.m('A6','POST','NUS'))}、Validity={pct(self.m('A6','POST','Validity'))}，三组一致。完整其余指标见表。
5. 在线 guidance 的独立价值是什么？它定义并验证了一种生成期间接入力信号的实现路径，Formal256 上相对无该引导的基线有效且跨势支持。但当前数据没有证明其在同预算下优于简单 POST，也没有证明“改变生成轨迹”必然更有益。若仅优化当前 MatterSim MaxF，POST 是更强的实证基线。

复用已有 A6（745000–745031）。这些 seed 在 A6 注册前通过全 worktree 历史扫描，和其他正式、P0、消融 cohort 不重叠。本轮是同一要求的归档与统计补齐，不以“再换一组新 seed”制造额外证据。POST 使用同一个 C0 文件且 SHA256 一致，每种修正方法每样本实际调用 20 次，失败/重试计入；本批失败调用为零。最终评分/松弛属于三组统一评价成本，与生成修正预算分列。'''
        r=self.old['A5']
        # Derive the retention values from the actual four arms, not prose.
        magc=self.m('A5','C0','Property MAE');forcec=self.m('A5','C0','MaxF')
        retmag=(magc-self.m('A5','AB','Property MAE'))/(magc-self.m('A5','A0','Property MAE'))
        retforce=(forcec-self.m('A5','AB','MaxF'))/(forcec-self.m('A5','B0','MaxF'))
        np.testing.assert_allclose([retmag,retforce],[r['retention_CFG'],r['retention_force']],atol=1e-12)
        self.interaction=self.mt('A5')+f'''
ADAPTIVE_CFG_PLUS_F0=NOT_COMPATIBLE，限定为“未通过冻结双收益保留率要求”，不等于所有指标全无改善。原 A5 执行完成、审计 PASS，科学判定 FAIL。

定义属性保留率为 (MAE_C0-MAE_AB)/(MAE_C0-MAE_A0)，力保留率为 (MaxF_C0-MaxF_AB)/(MaxF_C0-MaxF_B0)。本批分别 {pct(retmag)} 和 {pct(retforce)}；后一项低于冻结 70% 要求。AB 相对 C0 的属性、NUS 和 MaxF 仍有点估计改善，不能把组合负结果描述成“全面失败”。

A0 单独改善本批磁密度 MAE，但 MatterSim MaxF 均值上升；B0 单独降低 MaxF。两个分支作用目标不同。结果提示相互作用或耦合，但未完成因果分解。最终论文采用从 MatterGen 分出的两条独立方法主线，不要求串联成组合最终模型，不开展 rescue tuning。'''
        gate=self.old['E3_gate'];primary=self.old['E3_primary']['E3-G']
        self.e3=self.mt('E3 Formal256')+f'''
E3-G role=STRONG POST-GENERATION BASELINE / FALLBACK，不作为第二创新主线。其阶段是“生成完成后，learned gate 决策，再作基于 CHGNet 力的局部位置精修”，而 RC-NFGD 在最后一段 reverse diffusion 内使用 MatterSim 力修正位置 score 后继续采样。两者的力源及调用定义并不相同；仅 A6 的 POST 与 F0 构成同一 MatterSim 实际调用预算的直接比较。

E3-G 相对本 cohort 的 C0 预松弛 MaxF 改善 {pct(-primary['relative_change'])}；原始绝对差 E3-G-C0={fmt(primary['mean_difference'])} eV/Å，95% CI [{fmt(primary['bootstrap_95_ci'][0])}, {fmt(primary['bootstrap_95_ci'][1])}] eV/Å；原始严格符号 wins/ties/losses={primary['wins']}/{primary['ties']}/{primary['losses']}，Holm p={fmt(primary['wilcoxon_p_holm'])}。这是绝对差 CI，不得误标成相对百分比 CI。

Gate 启用率={pct(gate['behavior']['E3-G']['refinement_rate'])}，启用数={gate['behavior']['E3-G']['refinement_count']}/256；平均接受步数={fmt(gate['behavior']['E3-G']['accepted_steps_mean'])}。Always-on harm rate={pct(gate['e3a_harm_rate'])}，gated harm rate={pct(gate['e3g_harm_rate'])}。Harm 采用 1e-6 eV/Å 容忍阈值，而原始主表 wins 使用严格符号，不能混同。接受步数不是全部原子势实际调用数。

E3-G cohort=40000–40255、target=0.1；F0 Formal256 cohort=740000–740255、target=0.2。即便生成 checkpoint 一样，cohort 和条件目标不同也不允许绝对 MaxF 排名；相对改善仅是各自配对基线下的描述，不是直接统计优越比较。归档仅有 relaxation elapsed，不能当成端到端生成+精修用时。历史 Mag MAE/Hit/原子平均力/端到端时间缺项标为 NA。'''
        self.statsnote='''配对单位为 seed，所有注册 seed 都保留。连续指标以“method-baseline”的差及其 95% CI 报告；较小更好的指标，相对改善=(baseline均值-method均值)/baseline均值。bootstrap 每次一起重采样配对，再计算均值比，而非平均逐 seed 百分比。比例类同时给 fraction 差，可乘 100 表示 pp。

历史正式检验不被覆盖：P0 bootstrap seed=20260913，Formal32=20260914，A1/A3/A4/A5/A6及其独立势=20260915，E3=20260728；这些已记录流程均为 20,000 次。I1 原始报告的随机种子/次数没有完整归档，本轮保留其原始 CI/p，不臆造参数。另行提供可复现的补充 20,000 次 paired percentile bootstrap（seed=20260915），标注为 retrospective/supplementary，不能替换原始冻结判定。

补充区间未做多重比较校正。NUS/Unique 的 bootstrap 固定原 cohort 标签，不在重复采样的集合上重跑去重，因此区间是条件于本批定义的标签；不要外推成全生成分布的唯一性保证。Wins 默认容忍差 1e-12；E3 历史严格符号统计和 harm 阈值另行保留。不存在看结果后更换 evaluator、删掉失败样本或改 primary endpoint 的操作。'''
        self.efficiency='''Adaptive CFG 复用条件/无条件神经网络输出，仅增加按字段归约和两个标量 EMA 状态。额外归约计算随得分张量总元素数增长，控制器常驻状态为 O(1)；并不新增训练参数或神经网络前向，但 GPU→CPU 标量同步可能产生实际开销，不能把“无额外 forward”写成“零开销”。

RC-NFGD 在 1000-step reverse diffusion 的最后 20 个 predictor 增加 20 次 MatterSim 力计算；相对采样工作量为 T_base+20*T_potential+映射/安全检查开销，不把原子势计算成本当作常数。正式 n=256 的真实调用总数为 5120，失败/回退为零。峰值显存是每个采样段的 PyTorch allocated 统计，不是 GPU 总显存，也不代表训练成本。

已有运行时间来自共享服务器。P0 和 A5 有明确其他任务竞争记录，Formal256 各组运行顺序也不是随机交叉的隔离微基准。不得从 F0 偶尔稍快推出加速结论，不剔除慢 seed，不比较不同 cohort 的绝对运行时。表中 Runtime 为记录的采样/势加载范围，不包含离线评价、MatterSim 松弛或 reference 构建。E3 的松弛耗时另列；其生成+精修时间缺失。
'''+table(self.main[['Cohort','Method','n','Runtime','Runtime scope']])
        self.limits='''1. 创新点一的 Formal256 满足历史方向性门槛，但三个主指标的 CI 均跨零；不能写成普遍或显著优越。其历史目标属性没有独立预测验证。
2. 新增 n=8 组件消融只补充机制诊断，统计效能有限；无法证明每个组件在所有任务上必要。逐字段 RMS 再等权平均并不等于跨字段尺度标准化，数值较大的字段仍可能主导统计反馈。
3. 仅覆盖一个条件 checkpoint、有限条件目标和数据域；不能保证跨模型/跨材料域可迁移。
4. MatterSim 与 CHGNet 都是代理，可能存在训练数据重叠。CHGNet 也参与属性护栏，不是完全未使用过的盲测模型。
5. 没有真实 DFT、声子、实验合成或动力学稳定性验证；几何 Validity 不是化学有效率或可合成性。
6. F0 的收益不意味着 equal-budget POST 更差；MatterSim 主指标上 POST 更好，独立势直接对比尚无明确胜者。
7. 有界修正必要性不受当前消融支持；简单组合未达到双收益保留要求；F1 未取得需要替换 F0 的收益。
8. 方向对照重放参考轨迹幅度，anti-force 非自身轨迹实时反梯度；未做 clean-x0 与 noisy-x_t 的严格等预算对照。
9. 不同 cohort/目标/运行资源不适合直接绝对排名，运行时间有共享 GPU 竞争混杂。
10. 原始 I1 部分字段和原 bootstrap RNG/次数未归档，E3 端到端运行时缺失；NA 和缺失元数据被明确保留。补充重算只扩大可审计性，不抹去历史来源局限。'''

    def existing_reports(self):
        self.contents()
        reports={'equal_budget_online_vs_post':('EXP-01 等预算在线与后处理',self.online,'A6'),
          'final_independent_mlip_validation':('EXP-02 独立势 Formal256 验证',self.independent,'A1'),
          'final_force_direction_audit':('EXP-04 力方向消融',self.direction,'A3'),
          'final_bounded_correction_audit':('EXP-05 有界修正消融',self.bound,'A4'),
          'final_interaction_analysis':('EXP-06 两创新相互作用',self.interaction,'A5'),
          'final_e3g_baseline_audit':('EXP-08 E3-G 强后处理基线',self.e3,'E3 Formal256')}
        for folder,(title,body,cohort) in reports.items():
            out=PROJECT/'experiments'/folder
            write(out/'final_report.md',f'# {title}\n\n执行状态：COMPLETED_EXISTING；只补充审计和统计，不重复生成。\n\n'+body+'\n\n'+self.provenance)
            self.raw[self.raw.cohort==cohort].to_csv(out/'per_structure_metrics.csv',index=False,float_format='%.17g',na_rep='NA')
            self.stats[self.stats.cohort==cohort].to_csv(out/'paired_statistics.csv',index=False,float_format='%.17g',na_rep='NA')
            self.dist[self.dist.Cohort==cohort].to_csv(out/'distribution_statistics.csv',index=False,float_format='%.17g')
        out=PROJECT/'experiments/equal_budget_online_vs_post'
        source=F0/'experiments/mattersim_equal_budget_post_generation'
        for name in ['seeds.json','config.yaml']:
            path=out/name
            if not path.exists():path.symlink_to(source/name)
        write(ROOT/'statistical_results.md','# 最终统计口径及结果\n\n'+self.statsnote+'\n\n## 创新点一原始正式统计\n\n'+self.i1results+'\n\n## F0 正式主结果\n\n'+self.formal+'\n\n## 独立评价\n\n'+self.independent+'\n\n其余全部配对比较见 paired_statistics.csv。\n\n'+self.provenance)
        write(ROOT/'efficiency_results.md','# 效率及计算成本\n\n'+self.efficiency+'\n\n'+self.provenance)
        efficiency=[]
        for (cohort,method),g in self.raw.groupby(['cohort','method'],sort=False):
            available=g.Runtime.notna().all()
            r={'Cohort':cohort,'Method':method,'n':len(g),'runtime_mean_s':g.Runtime.mean() if available else np.nan,
               'runtime_median_s':g.Runtime.median() if available else np.nan,'runtime_p95_s':g.Runtime.quantile(.95) if available else np.nan}
            for col,operation in [('actual_mattersim_calls','mean'),('failed_mattersim_calls','sum'),('peak_allocated_bytes','max'),('mattergen_score_calls','mean')]:
                r[col+'_'+operation]=getattr(g[col],operation)() if col in g and g[col].notna().all() else np.nan
            efficiency.append(r)
        ef=pd.DataFrame(efficiency);ef.to_csv(ROOT/'efficiency_results.csv',index=False,float_format='%.17g',na_rep='NA')
        with (ROOT/'efficiency_results.md').open('a') as f:f.write('\n\n## 调用及显存明细\n\n'+table(ef)+'\n字段缺失时不把 guidance event 或接受步数冒充 actual force-call count。\n')

    def final_materials(self):
        assert self.audit['Innovation1_minimal8']['complete'],'Do not generate final thesis chapters before minimal ablation evaluation completes.'
        traces=pd.read_csv(ROOT/'innovation1_trace_audit.csv')
        self.ablation1=self.mt('Innovation1 minimal8')+'\n\n'+self.st('Innovation1 minimal8',['E-hull','Stable','NUS','Property MAE'])+'\n\n'+table(traces.groupby('method',as_index=False).agg(events=('events','sum'),clipped_events=('clipped_events','sum'),fallback_events=('fallback_events','sum'),minimum_scale=('min_scale','min'),maximum_scale=('max_scale','max'),predictor_mean=('predictor_scale_mean','mean'),corrector_mean=('corrector_scale_mean','mean')))+'''
全部 8 个新配对 seed=746000–746007 在产生结果前注册。六组为 C0、完整 A0、仅 position 残差、移除 EMA 时间平滑(beta=0)、共享 predictor/corrector EMA、移除 multiplier/final 两层 clip。只作组件去除，不调其余参数；没有学习新模型。目标=0.1，和 I1 Formal256 一致，但该批不能并入正式256或补写为256属性验证。

机制轨迹逐行独立复算：2000 次 score 决策/样本，五个非 C0 组共 80,000 条，残差、EMA、ratio 和最终 scale 对应公式均匹配。去除 EMA 后 ratio 约为1，接近固定 CFG，但因 epsilon 并非位级严格相同。共享 EMA 保留原有 predictor/corrector 标签，但两阶段交替更新同一状态。POS_ONLY 仍向所有字段应用一个共享 scale。

表格展示所有方向，包括不利结果；不因 n=8 的波动改动 Formal256 冻结方法。即使完整 A0 不是每项指标最优，也不做参数回扫。Clip 是否在本批被触发以轨迹统计为准；无触发不构成未来必然稳定的证据。组件消融完成不等于每个组件的必要性均成立。'''
        mini_e=self.s('Innovation1 minimal8','C0','A0','E-hull')
        mini_p=self.s('Innovation1 minimal8','C0','A0','Property MAE')
        adverse=(f"新增批次没有复现原 Formal256 的质量改善方向：完整 A0 的 E-hull 从 {fmt(mini_e.baseline_mean)} 增至 {fmt(mini_e.method_mean)} eV/atom，"
          f"差值 {fmt(mini_e.delta_method_minus_baseline)}，补充未校正 bootstrap 95% CI [{fmt(mini_e.delta_ci95_low)}, {fmt(mini_e.delta_ci95_high)}]。"
          f"Stable 从 {pct(self.m('Innovation1 minimal8','C0','Stable'))} 降至 {pct(self.m('Innovation1 minimal8','A0','Stable'))}，"
          f"NUS 均为 {pct(self.m('Innovation1 minimal8','A0','NUS'))}；属性 MAE 从 {fmt(mini_p.baseline_mean)} 降至 {fmt(mini_p.method_mean)}，"
          f"其改善区间仍跨零。结果显示属性与结构质量的权衡，不能认定完整多字段/EMA 版本在本批全部主指标上优于简化版本。"
          "n=8 的 percentile bootstrap 和离散比例区间统计效能/覆盖率均有限，不将其单独升级为普遍显著性结论；同样不能因样本小而隐去负方向。"
          "创新点一继续保留为用户冻结的方法路线，但其效果证据必须标注为有限、跨批次不一致，而不是稳定复现的普遍提升。")
        self.ablation1+='\n\n### 本次真实结果的解释\n\n'+adverse
        self.limits+='\n11. '+adverse
        write(PROJECT/'experiments/final_innovation1_audit/final_report.md',
          '# EXP-03 创新点一最终审计\n\n## 方法定义\n\n'+self.i1method+'\n\n## 正式结果与统计\n\n'+self.i1results+'\n\n## 最小组件消融\n\n'+self.ablation1+'\n\n## 效率\n\n'+self.efficiency+'\n\n## 局限\n\n'+self.limits+'\n\n'+self.provenance)
        algorithm1='''```text
输入：冻结条件生成模型、条件 c、同一配对 seed、1000 步采样配置
初始化 predictor_ema=None，corrector_ema=None
每次采样状态需要条件 score 时：
  联合计算 unconditional 和 conditional 得分
  对 atomic/pos/cell 分别计算残差 RMS，再取字段均值 d
  根据当前阶段选择对应 EMA，并用 d 更新
  计算 ratio、multiplier clip 和 scale clip
  对所有字段使用同一个 scale 执行 torch.lerp
  继续原 predictor/corrector 更新
输出：生成晶体；不新增参数训练
```'''
        algorithm2='''```text
输入：冻结 MatterGen、冻结 MatterSim、配对 seed、固定 CFG=2
每次官方得分计算后：
  若非 predictor 或 t>0.02：原样返回
  从原子 logits、位置 score、晶胞 score 构造完整 clean estimate
  在 clean 晶体上计算 MatterSim 原子力，累计真实势调用
  去除力的平移分量，按冻结 Fref 归一化并限制位移
  Cartesian -> fractional，检查周期候选的最小间距和有限性
  获取 ancestral predictor 实际 score 系数 c_score
  仅加入 delta_score_pos = delta_frac/c_score
  若检查失败：返回未加物理修正的原 score，保留失败记录
  继续原来的反向扩散，直到生成完成
输出：晶体、20事件轨迹、调用计数；评价成本另计
```'''
        setup='''统一基础模型为 official dft_mag_density checkpoint，SHA256 在 manifest 中按本地文件重新核对。固定 CFG 基线 scale=2，1000 diffusion steps，每步1次 corrector，batch_size=1，FP32、冻结参数、确定性 seed。原始 I1/E3 的条件目标为0.1，F0 主线及 A3/A4/A5/A6 为0.2，新增 I1 组件诊断为0.1。硬件为共享 NVIDIA H20；不同批次的并行与资源竞争记录分开。

原始 I1=256 pairs（20000–20255），F0 P0=16（710032–710047）、Formal32=32（730000–730031）、Formal256=256（740000–740255）。方向/有界/POST 分别为32 pairs，交互为64 pairs。两种方法并不共享所有正式 cohort，不拼接为一个大样本。'''
        metrics='''MaxF=单结构原子力模长最大值，再对结构取均值；Atomic Force=先在结构内平均原子力模长，再对结构等权平均。单位 eV/Å，均是 pre-relax 原结构评价。Independent 指 CHGNet 0.3.0 对相同原结构给出的力。

RMSD 是官方结构匹配流程中的松弛位移（Å）；E-hull 是 MatterSim 松弛能量相对冻结 Alex-MP 参考的代理量（eV/atom）；Stable 使用 0.1 eV/atom 阈值；NUS=Novel∧Unique∧Stable。各 cohort 内去重、匹配和分母保持原流程。Validity 此处为几何有效性，和 composition validity 单独区分。

CHGNet 磁密度代理为原子磁矩绝对值之和除以结构体积，MAE 对相应条件目标计算；Hit=绝对误差≤0.01。磁矩密度的物理单位可记 μB/Å³，旧代码字段以 a3 命名；不能将它称作真实 DFT 命中率。Runtime 与调用预算见独立效率表。'''
        sections3=[('3.1 Problem Definition',f'研究条件晶体逆向生成中的在线引导强度问题。{I1} 在不训练新网络的条件下，将当前扩散状态下的条件/无条件差异反馈到共享 CFG scale。该算法处理的是采样控制问题，不预设属性梯度或物理力场。'),
          ('3.2 Limitation of Fixed CFG','固定 scale=2 对所有时间和状态使用相同条件放大系数。残差大小随字段、时间和 predictor/corrector 阶段变化，固定强度本身不利用这些在线信息。这是方法动机；不能仅由动机推出目标属性误差必然下降。'),
          ('3.3 Multi-field Residual',self.i1method),
          ('3.4 Predictor / Corrector Online Statistics','两个阶段独立初始化和更新 EMA，防止交替更新的尺度差异直接混入同一参考。当前残差先进入 EMA 再用于 ratio；不使用训练集估计在线均值，不跨 seed 继承状态。参见 3.3 的 exact 公式与 3.10 的逐行轨迹审计。'),
          ('3.5 Adaptive Guidance Rule','ratio 相对当前阶段平滑参考偏大时，adaptive multiplier 随之增大；偏小时减小。两层 clip 与异常回退仅定义控制范围。所有字段仍共用一个标量，不采用 field-specific guidance，不额外启用分阶段 piecewise 调度。完整参数见 3.3。'),
          ('3.6 Algorithm',algorithm1),('3.7 Complexity',self.efficiency.split('RC-NFGD')[0]),
          ('3.8 Experimental Setup',setup),('3.9 Formal Results',self.i1results),('3.10 Ablation',self.ablation1),
          ('3.11 Discussion','最终保留该方法作为采样自适应控制主线；确认依据必须限定为历史预设方向性门槛，而非显著优越。新增8对结果没有复现其质量改善方向，不能称为跨批次稳定提升。该方法不需要和 RC-NFGD 串联。\n\n'+self.limits)]
        p0=self.effect('P0','B0','F0','MaxF')+'\n\n'+self.mt('P0')
        f32=self.effect('Formal32','C0','F0','MaxF')+'\n\n'+self.mt('Formal32')+'\n本段相对区间来自另行标注的最终补充 bootstrap；原 Formal32 seed=20260914 的区间保留在 original_decisions.json，不替换正式判定。'
        sections4=[('4.1 Motivation','对代理质量较好的生成晶体，仍可能存在较大局部原子受力。目标是在不训练新生成模型的前提下，将几何力信号用于最后一段反向扩散，并考察其是否改善原结构的预松弛受力。'),
          ('4.2 Limitation of Pure Data-driven Crystal Diffusion','现有 C0 生成结构的 MaxF 分布包含较大尾部。本实验据此研究额外力引导的收益；不把这一观察扩大为对所有数据驱动模型的理论否定，也不宣称 MaxF 降低等于热力学稳定。'),
          ('4.3 Late-stage Reliability Calibration',self.reliability),
          ('4.4 Clean-x0 Neural Force Evaluation',self.i2method),
          ('4.5 Periodic Coordinate Force Mapping','周期晶体位置用 fractional 坐标、神经势用 Cartesian 力。采用行向量约定 r=pL，按 Δp=Δr·inv(clean_cell) 转换，再除以实际 predictor score coefficient。modulo 操作用于候选安全检查；不能把 Cartesian 力直接加到 fractional score，也不能使用任意常数替代 predictor 系数。'),
          ('4.6 Bounded Force-guided Reverse Diffusion','修正只进入 position predictor score，之后继续采样；不直接改输出文件后宣称在线方法。0.005 Å 是该 clean-cell 映射中新增单事件贡献的最大值，不是最终整个晶体与基线的总距离。bound 的性能必要性见4.14的负结果。'),
          ('4.7 Algorithm',algorithm2),('4.8 Complexity and Runtime',self.efficiency),
          ('4.9 P0',p0),('4.10 Formal32',f32),('4.11 Formal256',self.formal),
          ('4.12 Independent MLIP Validation',self.independent),('4.13 Force-direction Ablation',self.direction),
          ('4.14 Bounded-correction Ablation',self.bound),('4.15 Online vs Equal-budget Post-processing',self.online),
          ('4.16 Comparison with E3-G',self.e3),('4.17 Discussion',self.interaction),('4.18 Limitations',self.limits)]
        manifest=js(ROOT/'reproducibility_manifest.json')
        ckpt=table(pd.DataFrame([{'Checkpoint':p,'SHA256':h} for p,h in manifest['checkpoint_sha256'].items()]))
        sections5=[('5.1 Experimental Setup',setup),('5.2 Dataset and Checkpoints','使用仓库冻结 checkpoint 与 Alex-MP reference_MP2020correction.gz。参考文件与所有可追溯脚本见 manifest；没有新训练。\n\n'+ckpt),
          ('5.3 Evaluation Metrics',metrics+'\n\n'+self.statsnote),
          ('5.4 Baselines','MatterGen C0 是每个 cohort 内对应基线；A0=Adaptive CFG；F0/B0=RC-NFGD（B0只在A5表示F0，而P0的B0是官方基线，以cohort为准）；E3-G 为强后处理基线；POST 是本轮等预算问题对应的简单 bounded force descent；G/T 前缀为方向/有界消融。不要跨 cohort 混淆相同短标签。'),
          ('5.5 Innovation1 Results',self.i1results),('5.6 Innovation2 Results',self.formal),
          ('5.7 Independent Evaluator Results',self.independent),('5.8 Ablation',self.ablation1+'\n\n'+self.direction+'\n\n'+self.bound),
          ('5.9 E3-G Baseline',self.e3),('5.10 Online vs Post-generation Comparison',self.online),
          ('5.11 Interaction Between Innovations',self.interaction),('5.12 Efficiency',self.efficiency),
          ('5.13 DFT Verification if available',(PROJECT/'experiments/dft_verification/environment_audit.md').read_text()),
          ('5.14 Limitations',self.limits)]
        for filename,title,sections in [('chapter3_innovation1.md','第3章 '+I1,sections3),('chapter4_innovation2.md','第4章 '+I2,sections4),('chapter5_experiments.md','第5章 实验设置与结果',sections5)]:
            figures={'chapter3_innovation1.md':['adaptive_cfg_pipeline','innovation1_formal_results','innovation1_minimal_ablation'],
                'chapter4_innovation2.md':['rc_nfgd_pipeline','formal256_maxf_paired','formal256_maxf_distribution','formal256_tail_metrics','independent_mlip_validation','force_direction_ablation','bounded_correction_ablation','online_vs_post'],
                'chapter5_experiments.md':['overall_method_framework','interaction_analysis']}[filename]
            figure_links='关联图：'+ '；'.join(f'[{name}]({ROOT}/plots/{name}.png)' for name in figures)+f'。图题与边界见 [plot_captions.md]({ROOT}/plot_captions.md)。'
            write(PROJECT/'thesis/final_materials'/filename,'# '+title+'\n\n'+TITLE+'\n\n这是依据已完成实验自动生成的写作素材，文中不新增未经实验验证的结论。\n\n'+self.provenance+'\n\n'+figure_links+'\n\n'+'\n\n'.join('## '+h+'\n\n'+body for h,body in sections))
        questions=[('Innovation1 解决 MatterGen 的什么问题？','固定 CFG 不利用当前扩散状态的条件残差反馈；本方法使共享引导强度在线自适应。Formal256 的目标属性本身未验证，因此不能写成已证明属性控制误差显著降低。'),
          ('Innovation1 的核心算法是什么？',self.i1method),('Innovation1 的 Formal 结果是什么？',self.i1results+'\n\n'+adverse),
          ('Innovation2 解决什么问题？','针对原始生成结构局部受力，在反向扩散末段接入冻结神经力信号，使原结构的代理几何质量改善。'+self.effect('A1','C0','F0','MaxF')),
          ('为什么选择 late-stage physics guidance？',self.reliability),
          ('为什么使用 clean-x0 而不是 noisy x_t？','使用模型的完整去噪晶体估计作为势函数输入，以降低强噪状态的不确定性。阶段诊断支持该实现；当前没有 noisy-x_t 的严格头对头消融，不能扩大成普遍优越结论。'),
          ('MatterSim force 如何映射进入 periodic crystal diffusion？',self.i2method),
          ('为什么 Innovation2 不是简单的 post-processing？','执行点位于 predictor 内，force->position-score 后还有 reverse diffusion 更新，区别于生成完成后修改最终晶体。但算法位置不同不自动意味着性能更优；等预算结果必须一起报告。'),
          ('Equal-budget POST baseline 的结果是什么？',self.online),
          ('为什么改善不是简单 MatterSim self-consistency？','未参与 force guidance 的冻结 CHGNet 在完整512结构上也显示配对平均力改善，支持跨势一致性。不能彻底排除共享训练数据及共同误差，也不构成 DFT 真值验证。'),
          ('Independent MLIP 是否支持？',self.independent),('真实 force direction 是否优于 random/shuffled/anti-force？',self.direction),
          ('Bounded correction 是核心机制还是稳定性设计？',self.bound),('E3-G 的角色是什么？',self.e3),
          ('为什么 Adaptive CFG 和 F0 不需要必须组合？',self.interaction),
          ('当前结果是否经过 DFT？','没有。已实际完成环境审计，未找到可运行 DFT 软件及真实赝势/工作流；完成 DFT 单点数为0。DFT_VERIFIED=False。'),
          ('当前有哪些 limitations？',self.limits)]
        write(ROOT/'final_report.md','# 论文最终实验与答辩问题\n\n'+TITLE+'\n\n'+self.provenance+'\n\n'+'\n\n'.join(f'## Q{i}. {q}\n\n{a}' for i,(q,a) in enumerate(questions,1)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    r=Reports();r.existing_reports()
    if not a.partial:r.final_materials()
    print('Existing reports updated.' if a.partial else 'Reports and all Chapter 3-5 materials generated from completed results.')
if __name__=='__main__':main()
