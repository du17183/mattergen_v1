# Local MatterGen A0 and clean-state code map

Base commit: `9bc6747a3ddfd26db6d931bcdb6df5d299844544`.

## Adaptive CFG

- `mattergen/diffusion/sampling/classifier_free_guidance.py`
  - `GuidedPredictorCorrector._score_fn`: builds the joint conditional and
    unconditional batch, separates both scores, computes field residuals, and
    interpolates the A0 score.
  - `score_residual_rms`: independent RMS summaries for `cell`, `pos`, and
    `atomic_numbers`.
  - `_current_progress_and_phase`: reads predictor/corrector progress context.
- `mattergen/diffusion/sampling/guidance_schedule.py`
  - `GuidanceController.evaluate`: phase-local EMA and adaptive scale.
  - `_mean_valid_deltas`: current A0 scalar aggregation after field summaries.

## Predictor/corrector loop

- `mattergen/diffusion/sampling/pc_sampler.py`
  - `PredictorCorrector._sample_maybe_record`: prior draw and run-local hooks.
  - `PredictorCorrector._denoise`: complete corrector then predictor loop.
  - `_set_sampling_context`: sampling step, total steps, progress, and phase.
  - `_mask_replace`: installs sampled and mean fields without altering masks.
- `mattergen/diffusion/sampling/predictors_correctors.py`
  - `LangevinCorrector.step_given_score`: stochastic corrector update.
- `mattergen/diffusion/wrapped/wrapped_predictors_correctors.py`
  - `WrappedPredictorMixin.update_given_score`: periodic position wrapping.
- `mattergen/diffusion/d3pm/d3pm_predictors_correctors.py`
  - `D3PMAncestralSamplingPredictor.update_given_score`: atomic categorical
    transition from logits that represent the predicted clean distribution.

## Continuous clean-state recovery

- `mattergen/diffusion/corruption/sde_lib.py`
  - `SDE.marginal_prob` and VP/VE subclasses supply the marginal mean and
    standard deviation.
- For a score model under the local convention:
  `x0_hat = (x_t + std(t)^2 * score_t) / mean_coeff(t)`.
- Position uses the wrapped VE process and must be wrapped back into `[0,1)`.
- Cell uses a VP process and is converted independently from position.
- `mattergen/diffusion/model_utils.py::convert_model_out_to_score` confirms
  that the model output is converted to a true score before the sampler sees it.

## Atomic clean state

- Atomic numbers use D3PM, not the continuous formula.
- The physical oracle may use `argmax` of the predicted clean atomic logits only
  to construct a temporary CHGNet structure.
- RP-QTFG never writes this temporary atomic estimate back into the sample and
  never changes atomic logits.

## Planned insertion point

The MVP will preserve the exact A0 score and A0 sampler update as `safe_update`.
Physics guidance is evaluated after A0 conditional/unconditional fusion and
before the continuous position/cell update. Any invalid oracle result, residual
conflict, trust-region violation, or failed backtracking attempt returns the
unchanged A0 score/update.
