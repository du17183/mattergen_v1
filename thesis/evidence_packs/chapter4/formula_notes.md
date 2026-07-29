# 第4章公式与伪代码说明

## F4_RESIDUAL

$$r_{t,k}=s^{cond}_{t,k}-s^{uncond}_{t,k}$$

- 代码：`mattergen/diffusion/sampling/classifier_free_guidance.py::score_residual_rms` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：Residual remains field-specific until scalar RMS reduction.

## F4_FIELD_RMS

$$\delta_{t,k}=\sqrt{\operatorname{mean}(r_{t,k}^{\,2})}$$

- 代码：`mattergen/diffusion/sampling/classifier_free_guidance.py::score_residual_rms` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：Residual is cast to float before square/mean.

## F4_FIELD_MEAN

$$\delta_t=\frac{1}{|\mathcal K_t|}\sum_{k\in\mathcal K_t}\delta_{t,k}$$

- 代码：`mattergen/diffusion/sampling/guidance_schedule.py::_mean_valid_deltas` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：One scalar controls the common guidance scale; there are not three field scales.

## F4_EMA

$$m_{t,p}=\begin{cases}\delta_t,&m_{t-1,p}\ \mathrm{unset}\\ \beta m_{t-1,p}+(1-\beta)\delta_t,&\mathrm{otherwise}\end{cases}$$

- 代码：`mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：Predictor and corrector maintain independent EMA states.

## F4_RATIO

$$q_t=\frac{\delta_t}{m_{t,p}+\epsilon}$$

- 代码：`mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：Ratio is scalar.

## F4_MULTIPLIER

$$u_t=\operatorname{clip}\!\left(1+\alpha(q_t-1),0.25,4\right)$$

- 代码：`mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：Adaptive multiplier limit is distinct from final guidance limit.

## F4_GUIDANCE

$$g_t=\operatorname{clip}(g_0u_t,g_{\min},g_{\max})$$

- 代码：`mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：For schedule=adaptive, stage guidance equals base guidance g0.

## F4_CFG_FUSION

$$s_t^{CFG}=s_t^{uncond}+g_t(s_t^{cond}-s_t^{uncond})$$

- 代码：`mattergen/diffusion/sampling/classifier_free_guidance.py::GuidedPredictorCorrector._score_fn_unaccelerated` @ `5de00419eea2d8a9be303638f2db8ece15a22366`
- 性质：`exact`
- 说明：Implemented through torch.lerp(unconditional, conditional, g_t).

## 论文伪代码

```text
Input: x_t, t, base guidance g0; state m_predictor, m_corrector
1. Build unconditional and conditional inputs and execute the joint full-CFG model forward.
2. Split s_uncond and s_cond.
3. For k in {cell,pos,atomic_numbers}, compute residual r_k and scalar RMS delta_k.
4. Average valid field RMS values into scalar delta.
5. Select the EMA state belonging to the current predictor/corrector phase.
6. Initialize/update EMA, compute q, multiplier u and clipped shared guidance g.
7. For every corrupted field, return lerp(s_uncond, s_cond, g).
8. Continue the complete configured corrector/predictor update.
Output: guided score and updated phase-local EMA state.
```

明确：该算法不跳过Predictor、不跳过Corrector、不是Corrector Gating，也不是步数削减方法。
