# 第5章公式说明

## F5_STANDARDIZE

$$z_j=(x_j-\mu_j)/\sigma_j$$

- 代码：`research/postgen_fastgate/refiner_eval.py::build_network / StandardScaler` @ `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 性质：`interpreted`
- 说明：Mathematical summary of scikit-learn StandardScaler.

## F5_GATE_NETWORK

$$h=\tanh(W_1z+b_1),\qquad c=\sigma(W_2h+b_2)$$

- 代码：`research/postgen_fastgate/refiner_eval.py::build_network / MLPClassifier` @ `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 性质：`interpreted`
- 说明：Library-level mathematical interpretation; inference calls predict_proba.

## F5_PARAMETER_COUNT

$$14\times8+8+8\times1+1=129$$

- 代码：`research/postgen_fastgate/refiner_eval.py::train_gate network trainable_parameters` @ `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 性质：`exact`
- 说明：StandardScaler has no trainable neural-network parameters.

## F5_GATE_RULE

$$a=\mathbb 1[c\ge 0.5]$$

- 代码：`research/q3_frozen64.py::refine` @ `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 性质：`exact`
- 说明：Gate-off returns the original structure.

## F5_POSITION_PROPOSAL

$$\Delta x_i^{(b)}=\operatorname{clipnorm}(\eta\,2^{-b}F_i,\ R_{step}2^{-b})$$

- 代码：`research/postgen_fastgate/refiner_eval.py::position_proposal and advance` @ `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 性质：`exact`
- 说明：Positions are updated and wrapped; atomic numbers and cell are not updated.

## F5_ACCEPTANCE

$$\mathrm{accept}\iff \mathrm{finite\_safe}(X')\land E_{\mathrm{CHGNet}}(X')\le E_{\mathrm{CHGNet}}(X)+10^{-7}$$

- 代码：`research/postgen_fastgate/refiner_eval.py::finite_safe and advance` @ `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 性质：`exact`
- 说明：finite_safe also requires volume >0.1 and minimum distance >=0.5 angstrom.

## F5_TRUST_BOUND

$$\max_i\lVert x_i^{final}-x_i^{input}\rVert_{MIC}\le 5\times0.02=0.10\;\AA$$

- 代码：`research/q3_frozen64.py::run_refinement_subset and refine postcondition` @ `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 性质：`interpreted`
- 说明：Bound follows per-step caps and is explicitly checked after refinement; it is not a lattice optimization.
