TCL P0 FINAL:
GO

DML P0 FINAL:
FAIL

DFT_VERIFIED=False

# MatterGen TCL / DML independent P0 report

This is an 8-paired-seed P0 screen. It establishes an actionable signal, not a
formal significance claim. TCL and DML were implemented, trained, checkpointed,
generated, and evaluated independently. No TCL+DML model or experiment exists.

## 1–7. Reproducibility and scope

- Branch: `experiment/tcl-dml-p0`
- Parent commit: `70a85b70b2b06f63a7965a07d1520331f3e3e215`
- Final HEAD: reported by `git rev-parse HEAD` in the final handoff because a Git
  commit cannot contain its own hash.
- Initialization: official local `dft_mag_density` checkpoint; base checkpoint
  SHA256 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
- Historical global adapter, cross-field adapter, and quality adapter were all
  disabled. The MatterGen model used standard full finetuning: every 48,760,443
  pretrained model parameter was trainable in FT0, TCL, and DML.
- To make the complete named optimizer scope identical, the same 83-parameter
  `NoiseAwareFieldScheduler` namespace was instantiated in every trained arm.
  Total named scope was 48,760,526 parameters in all arms. It received gradients
  only in DML (4 scheduler tensors at step 1); FT0/TCL had zero scheduler tensors
  with gradients. All three had 213 finite-gradient model tensors at step 1.
- Trainable-name SHA256 was identical in all arms:
  `0c9d6e6efdb196fc5df547686e1148985e157825140a2375d0da819c672b4198`.
- The exact previous 1024-train/128-validation cached split was reused from
  `experiments/global_transformer_adapter_p0/data/cache`; selection seed was
  20260907 and training seed was 20260908.

Independent checkpoint SHA256 values:

| Arm | Checkpoint SHA256 |
|---|---|
| FT0 | `1cd254568463ae4d7774193fa0a0e97925cdad3b6c505c1e5f5dd4e7968f6960` |
| TCL | `8dd9879f11edea7f25f587ae72f928a6f3ab2e5c81fc9400c13642364ad90443` |
| DML | `9d9a10f3200e9c8cbd74fea69e59e1192b1e12727a1d50af72d52362b7abe340` |

## 8–12. Fair two-view training and FT0

For every clean structure, two timesteps were drawn through MatterGen's original
timestep sampler. The second was redrawn wherever
`abs(t1-t2) < 0.2*T`; the pair was then sorted into `t_low,t_high`. Thus the
normalized gap was at least 0.2 while preserving the original sampler subject to
that constraint.

Position and cell used the same Gaussian noise realization in both views. Cell
noise was symmetrized with MatterGen's variance-preserving helper. Atomic-number
D3PM corruptions were independently drawn for the two timesteps, because a shared
categorical noise tensor is not the correct D3PM coupling.

FT0 used no consistency and no dynamic reweighting. Its loss was the mean of the
two original per-structure MatterGen mixed-field objectives:

`L_FT0 = 0.5 * sum_{v in {low,high}} (1.0 L_atomic,v + 0.1 L_pos,v + 1.0 L_cell,v)`.

The field losses themselves, including D3PM hybrid coefficient 0.01 and original
reductions, were not redefined.

## 13–19. TCL definition

The low-noise view is the stop-gradient teacher. MatterGen outputs are first
converted into estimates of the common clean state; raw scores from different
timesteps are never directly compared.

- Atomic: the D3PM head's logits represent `p_theta(x0 | xt)`, so atomic
  consistency is implemented as
  `KL(stopgrad(softmax(logits_low)) || softmax(logits_high))`, averaged per atom
  and then per structure. Atomic consistency is therefore present.
- Position: with MatterGen's score-times-std parameterization,
  `x0_hat = wrap(xt + sigma_t * output_t)`. The loss is the mean squared periodic
  minimum-image displacement
  `||wrap_delta(x0_hat_high - stopgrad(x0_hat_low))||^2` in fractional coordinates.
- Cell: for `xt = alpha_t*x0 + (1-alpha_t)*mu_inf + sigma_t*z`, first recover
  `x0_hat = (xt + sigma_t*output_t - (1-alpha_t)*mu_inf)/alpha_t`, then standardize
  it as `(x0_hat-mu_inf)/sqrt(var_inf)`. Cell consistency is the mean squared
  difference between the two standardized clean-cell estimates.

The three consistency components are averaged equally. The training objective is
`L_TCL = L_FT0 + lambda_cons * mean(L_atomic_cons,L_pos_cons,L_cell_cons)`.
`lambda_cons` increases linearly from 0 at step 1 to 0.1 at step 100 and remains
0.1 thereafter.

## 20–27. DML definition and learned routing

Original field weights were Atomic=1.0, Position=0.1, Cell=1.0. The scheduler input
was normalized timestep `t in [0,1]`; its network was
`Linear(1,16) -> SiLU -> Linear(16,3)`, with a zero-initialized final layer.
Multipliers were `r_k=1+0.5*tanh(logit_k)`, strictly bounded to (0.5,1.5).

For `S0=sum_k base_k`, the effective weights were
`w_k = S0*(base_k*r_k)/sum_j(base_j*r_j)`. Each view had independent weights, and
they were applied to each structure's three field losses before batch reduction.
The multiplier-to-one anchor coefficient was 0.01. DML used no consistency loss.

| Field | min | mean | max | t=0–.2 | .2–.4 | .4–.6 | .6–.8 | .8–1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Atomic | 0.95746 | 1.03015 | 1.08241 | 1.05050 | 1.04025 | 1.03042 | 1.01986 | 1.00952 |
| Position | 0.67896 | 0.87050 | 1.00000 | 0.89495 | 0.88317 | 0.86994 | 0.85681 | 0.84724 |
| Cell | 1.00000 | 1.12340 | 1.30475 | 1.09956 | 1.11103 | 1.12391 | 1.13671 | 1.14619 |

Boundary fraction was 0 for every field; no weight collapse occurred. The learned
routing progressively downweighted position and upweighted cell as noise rose.

## 28–30. Real training and fixed-loss validation

All arms used AdamW, learning rate 1e-4, weight decay 1e-4, effective batch 16,
1000 steps, and two views per clean structure. All losses and checked gradients
remained finite.

| Arm | final total | final base | method term | first/last-50 base | in-training original val | runtime s | peak MiB |
|---|---:|---:|---|---:|---:|---:|---:|
| FT0 | 0.274908 | 0.274908 | none | 0.298493 / 0.295913 | 0.319830 | 155.35 | 10950.03 |
| TCL | 0.445596 | 0.294727 | consistency=1.508692 at lambda=0.1 | 0.297968 / 0.295808 | 0.310160 | 163.01 | 10950.03 |
| DML | 0.211299 | 0.272624 | dynamic=0.210835, anchor=0.000464 | 0.297684 / 0.292829 | 0.337136 | 157.50 | 10950.04 |

TCL's final consistency components were Atomic=1.727239,
Position=0.067637, Cell=2.731199. Final per-field base losses were:
FT0=(0.108721,1.567336,0.009453),
TCL=(0.110245,1.708790,0.013603), and
DML=(0.110450,1.505009,0.011673) for Atomic/Position/Cell.

The independent 128-structure offline evaluation used corruption seed 20260929
and only the original fixed MatterGen loss:

| Arm | total | Atomic | Position | Cell | delta vs C0 | delta vs FT0 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.263821 | 0.108159 | 1.466679 | 0.008995 | 0 | -0.051517 |
| FT0 | 0.315338 | 0.127197 | 1.673274 | 0.020814 | +0.051517 | 0 |
| TCL | 0.304887 | 0.122021 | 1.625876 | 0.020279 | +0.041066 | -0.010451 |
| DML | 0.323964 | 0.130305 | 1.740998 | 0.019559 | +0.060143 | +0.008626 |

This static validation loss was not used as a substitute for generation quality.

## 31–45. Real P0 generation, MatterSim, quality, and tails

Paired seeds were 80000–80007. The four arms were C0, FT0, TCL, and DML. Each arm
generated 8/8 structures successfully with the official 1000-step predictor/
corrector sampler, one corrector step per time, target `dft_mag_density=0.1`,
constant CFG=2, and batch size 1. Mean generation times were C0=116.17 s,
FT0=117.66 s, TCL=117.06 s, DML=116.70 s.

MatterSim-5M relaxation at `fmax=0.05 eV/A` succeeded for 8/8 structures in every
arm. The frozen Alex-MP/MP2020-correction reference was used for official metrics.

| Arm | E-hull mean / median / max (eV/atom) | Stable | NUS | Novel | Unique |
|---|---|---:|---:|---:|---:|
| C0 | 0.106113 / 0.119131 / 0.255537 | 0.375 | 0.375 | 1.000 | 1.000 |
| FT0 | 0.109998 / 0.089232 / 0.229996 | 0.625 | 0.125 | 0.500 | 1.000 |
| TCL | 0.100462 / 0.093270 / 0.180096 | 0.625 | 0.375 | 0.750 | 1.000 |
| DML | 0.167630 / 0.136477 / 0.325196 | 0.375 | 0.125 | 0.750 | 1.000 |

| Arm | RMSD mean / median / P95 / max (A) | Atomic-force mean (eV/A) | Structure max-force mean / P95 / max (eV/A) |
|---|---|---:|---|
| C0 | 0.157224 / 0.083293 / 0.457498 / 0.491945 | 0.236355 | 0.479674 / 1.086605 / 1.360870 |
| FT0 | 0.053439 / 0.047141 / 0.110630 / 0.113372 | 0.186033 | 0.435060 / 0.961095 / 1.008138 |
| TCL | 0.033702 / 0.020092 / 0.077147 / 0.081808 | 0.102338 | 0.192651 / 0.412570 / 0.437217 |
| DML | 0.053179 / 0.038984 / 0.140199 / 0.163680 | 0.223993 | 0.422268 / 0.851547 / 0.906693 |

| Arm | Relaxation steps mean / median / P95 / max | >100 | >200 | >400 |
|---|---|---:|---:|---:|
| C0 | 112.000 / 52.5 / 336.30 / 372 | 2 | 2 | 0 |
| FT0 | 45.125 / 37.5 / 70.45 / 75 | 0 | 0 | 0 |
| TCL | 36.000 / 32.0 / 56.55 / 59 | 0 | 0 | 0 |
| DML | 50.250 / 37.0 / 111.50 / 122 | 1 | 0 | 0 |

Severe outliers under the requested thresholds were retained:

| Method | seed | formula | E-hull | RMSD A | max-force eV/A | relax steps | flag |
|---|---:|---|---:|---:|---:|---:|---|
| C0 | 80001 | Eu6Ga2As4PO | 0.255537 | 0.491945 | 0.446107 | 270 | relaxation >200 |
| C0 | 80002 | EuTeO4 | -0.017631 | 0.393524 | 0.577255 | 372 | relaxation >200 |
| C0 | 80005 | Eu3OF3 | 0.132958 | 0.117629 | 1.360870 | 40 | max-force >1 |
| FT0 | 80001 | EuMn7(AsP2)2 | 0.199936 | 0.092783 | 1.008138 | 75 | max-force >1 |

There were no RMSD >0.5 A and no relaxation >400 cases. TCL and DML had no
requested severe outliers. `tail_analysis.csv` additionally retains softer
E-hull >0.1 and relaxation >100 flags rather than deleting them.

## 46–52. Scientific comparisons

FT0 vs C0: continued full finetuning was not a neutral control. It improved
Stable by 25 percentage points and strongly improved RMSD, forces, and relaxation,
but mean E-hull worsened by 0.00388 eV/atom, NUS fell by 25 points, and Novel fell
by 50 points. It traded diversity/novelty for geometry and stable fraction.

TCL vs C0: mean E-hull improved by 0.00565 eV/atom, Stable rose by 25 points,
NUS was unchanged, Unique was unchanged, and Novel fell by 25 points. Geometry
improved strongly: RMSD -0.12352 A, atomic force -0.13402 eV/A, structure max-force
-0.28702 eV/A, and relaxation -76 steps on average. The paired favorable counts
were 4/8 for E-hull, 3/8 Stable (4 ties), 5/8 RMSD, and 6/8 for both force measures.

TCL vs FT0, the primary comparison: mean E-hull improved by 0.00954 eV/atom;
Stable remained 0.625; NUS and Novel each rose by 25 points; Unique remained 1.0.
RMSD improved by 0.01974 A, atomic force by 0.08370 eV/A, max force by
0.24241 eV/A, and relaxation by 9.125 steps. Paired favorable/unfavorable/tie
counts were E-hull 4/4/0, NUS 3/1/4, Novel 3/1/4, RMSD 5/3/0,
atomic force 6/2/0, max force 6/2/0, and steps 5/3/0. Leave-one-out signs were
robust for NUS, Novel, RMSD, both force metrics, and relaxation, but not E-hull.
TCL therefore provides multi-dimensional benefit beyond ordinary finetuning.

DML vs C0: mean E-hull worsened by 0.06152 eV/atom (only 1/8 favorable), Stable
was unchanged, NUS and Novel each fell by 25 points, and Unique was unchanged.
RMSD and relaxation improved relative to C0, but this does not offset the marked
thermodynamic-quality loss.

DML vs FT0, the primary comparison: mean E-hull worsened by 0.05763 eV/atom
(3 favorable, 5 unfavorable; worsening leave-one-out robust), Stable fell by
25 points, NUS was unchanged, Novel rose by 25 points, and Unique was unchanged.
RMSD was essentially tied (-0.00026 A), atomic force worsened by 0.03796 eV/A,
max force improved only 0.01279 eV/A without leave-one-out robustness, and
relaxation worsened by 5.125 steps with a new >100-step case. Thus DML does not
provide additional overall value beyond ordinary finetuning despite non-collapsed
routing.

## 53–58. Decisions and exactly one next step

TCL P0 FINAL: **GO**. It meets the preregistered pattern E-hull down plus
RMSD/force down versus FT0, while Novel improves, Unique is preserved, and no new
severe geometry/relaxation tail appears. This is still an 8-seed P0 signal.

DML P0 FINAL: **FAIL**. It materially worsens E-hull and Stable versus FT0 and
does not improve NUS; its Novel gain and non-collapsed routing are insufficient.

Only TCL deserves the later 32-seed validation. The one recommended next step is:
run a fresh paired 32-seed C0/FT0/TCL validation with the same frozen generation
and MatterSim pipeline, without DML, without TCL+DML, and without any method
change. That next experiment was not started in this branch.
