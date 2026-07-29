# Formula Registry

| formula_id | 章 | 公式 | 源码符号 | commit | 性质 | 人工确认 |
| --- | --- | --- | --- | --- | --- | --- |
| F3_MAX_FORCE | 3 | $F_{\max}=\max_i\lVert\mathbf F_i\rVert_2$ | `research/q3_frozen64.py::relax worker pre_relax_max_force_ev_ang` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | exact | False |
| F3_RMSD | 3 | $\operatorname{RMSD}=\operatorname{MatcherRMSD}(X_{\mathrm{relaxed}},X_{\mathrm{initial}})\;[\AA]$ | `mattergen/evaluation/utils/utils.py::compute_rmsd_angstrom` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | interpreted | False |
| F3_STABLE | 3 | $\mathrm{Stable}=\mathbb 1[E_{\mathrm{hull}}\le 0.1\;\mathrm{eV/atom}]$ | `mattergen/evaluation/metrics/energy.py::EnergyCapability.is_stable` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | exact | False |
| F3_NUS | 3 | $\mathrm{NUS}=\mathrm{Novel}\land\mathrm{Unique}\land\mathrm{Stable}$ | `mattergen/evaluation/metrics/energy.py::FracNovelUniqueStableStructures.compute_pre_aggregation_values` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | exact | False |
| F3_HARM | 3 | $\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]$ | `research/q3_formal256.py::FORCE_HARM_EPSILON and gate mechanism analysis` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | exact | False |
| F4_RESIDUAL | 4 | $r_{t,k}=s^{cond}_{t,k}-s^{uncond}_{t,k}$ | `mattergen/diffusion/sampling/classifier_free_guidance.py::score_residual_rms` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F4_FIELD_RMS | 4 | $\delta_{t,k}=\sqrt{\operatorname{mean}(r_{t,k}^{\,2})}$ | `mattergen/diffusion/sampling/classifier_free_guidance.py::score_residual_rms` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F4_FIELD_MEAN | 4 | $\delta_t=\frac{1}{|\mathcal K_t|}\sum_{k\in\mathcal K_t}\delta_{t,k}$ | `mattergen/diffusion/sampling/guidance_schedule.py::_mean_valid_deltas` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F4_EMA | 4 | $m_{t,p}=\begin{cases}\delta_t,&m_{t-1,p}\ \mathrm{unset}\\ \beta m_{t-1,p}+(1-\beta)\delta_t,&\mathrm{otherwise}\end{cases}$ | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F4_RATIO | 4 | $q_t=\frac{\delta_t}{m_{t,p}+\epsilon}$ | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F4_MULTIPLIER | 4 | $u_t=\operatorname{clip}\!\left(1+\alpha(q_t-1),0.25,4\right)$ | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F4_GUIDANCE | 4 | $g_t=\operatorname{clip}(g_0u_t,g_{\min},g_{\max})$ | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F4_CFG_FUSION | 4 | $s_t^{CFG}=s_t^{uncond}+g_t(s_t^{cond}-s_t^{uncond})$ | `mattergen/diffusion/sampling/classifier_free_guidance.py::GuidedPredictorCorrector._score_fn_unaccelerated` | `5de00419eea2d8a9be303638f2db8ece15a22366` | exact | False |
| F5_STANDARDIZE | 5 | $z_j=(x_j-\mu_j)/\sigma_j$ | `research/postgen_fastgate/refiner_eval.py::build_network / StandardScaler` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | interpreted | False |
| F5_GATE_NETWORK | 5 | $h=\tanh(W_1z+b_1),\qquad c=\sigma(W_2h+b_2)$ | `research/postgen_fastgate/refiner_eval.py::build_network / MLPClassifier` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | interpreted | False |
| F5_PARAMETER_COUNT | 5 | $14\times8+8+8\times1+1=129$ | `research/postgen_fastgate/refiner_eval.py::train_gate network trainable_parameters` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | exact | False |
| F5_GATE_RULE | 5 | $a=\mathbb 1[c\ge 0.5]$ | `research/q3_frozen64.py::refine` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | exact | False |
| F5_POSITION_PROPOSAL | 5 | $\Delta x_i^{(b)}=\operatorname{clipnorm}(\eta\,2^{-b}F_i,\ R_{step}2^{-b})$ | `research/postgen_fastgate/refiner_eval.py::position_proposal and advance` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | exact | False |
| F5_ACCEPTANCE | 5 | $\mathrm{accept}\iff \mathrm{finite\_safe}(X')\land E_{\mathrm{CHGNet}}(X')\le E_{\mathrm{CHGNet}}(X)+10^{-7}$ | `research/postgen_fastgate/refiner_eval.py::finite_safe and advance` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | exact | False |
| F5_TRUST_BOUND | 5 | $\max_i\lVert x_i^{final}-x_i^{input}\rVert_{MIC}\le 5\times0.02=0.10\;\AA$ | `research/q3_frozen64.py::run_refinement_subset and refine postcondition` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | interpreted | False |

`exact`仅表示与仓库代码数学等价；`interpreted`表示对库调用或实现流程的数学概括。
