# A0 + E3-G training-test leakage diagnostic

- Seeds `20000–20063`: overlap the frozen Q3 gate training set.
- Seeds `20064–20255`: not used for Q3 gate training.
- The overall 256 result is intentionally contaminated and is not a formal validation.
- A0 generation and completed A0 relaxation are reused exactly.
- No training, retuning, threshold change, or refinement change is permitted.
