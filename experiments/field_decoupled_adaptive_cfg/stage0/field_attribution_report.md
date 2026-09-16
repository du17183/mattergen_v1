# Stage 0 field-attribution diagnostic

Historical traces are used only to motivate and sanity-check field decoupling. They are not calibration, P0, or Formal evidence.

## Evidence boundary

- V4/V5 online traces show strongly heterogeneous conditional-unconditional residual magnitudes across atomic_numbers, position, and cell fields and across timesteps.
- The preceding stage-bounded P0 retained a property signal but failed physical guardrails; its status is hypothesis-generating only.
- Previous BOUNDED_PULSE_EFFECT: SUPPORTED; previous STAGE_SPECIFIC_EFFECT: NOT_SUPPORTED.
- Residual magnitudes have different units and tensor shapes, so raw magnitudes must not be interpreted as causal effect sizes.
- Causal field attribution is reserved for the new paired 12-policy calibration.

## Frozen diagnostic conclusion

All three score fields have nonzero, timestep-dependent conditioning residuals. A field-wise CFG implementation is therefore meaningful to test, but Stage0 does not preselect PROPERTY_FIELD or QUALITY_HARM_FIELD.

FIELD_ATTRIBUTION_DIAGNOSTIC = COMPLETE
