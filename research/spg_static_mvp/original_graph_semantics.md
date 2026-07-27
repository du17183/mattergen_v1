# Original MatterGen periodic graph semantics

This map is frozen against `main@9bc6747a3ddfd26db6d931bcdb6df5d299844544`
and PyTorch `2.7.1+cu128`. The official `dft_mag_density` checkpoint uses
`GemNetTCtrl`, which inherits graph construction, symmetric-edge reordering,
and triplet construction from `GemNetT`.

## Call path

| Stage | File and symbol | Input | Output |
|---|---|---|---|
| Score model | `mattergen/common/gemnet/gemnet_ctrl.py:69`, `GemNetTCtrl.forward` | fractional positions `[N,3]`, lattice `[B,3,3]`, atom types `[N]`, `num_atoms [B]` | score outputs |
| Interaction graph | `mattergen/common/gemnet/gemnet.py:515`, `GemNetT.generate_interaction_graph` | Cartesian positions `[N,3]`, lattice | ordered edges and triplets |
| PBC wrapper | `mattergen/common/utils/data_utils.py:231`, `radius_graph_pbc` | positions, lattice, `num_atoms`, cutoff | raw edge index, offsets, per-image count |
| OCP builder | `mattergen/common/utils/ocp_graph_utils.py:59`, `radius_graph_pbc` | same plus PBC flags | radius/top-50 graph |
| Symmetric reorder | `mattergen/common/gemnet/gemnet.py:421`, `GemNetT.reorder_symmetric_edges` | raw directed graph | symmetric ordered graph |
| Triplets | `mattergen/common/gemnet/gemnet.py:364`, `GemNetT.get_triplets` | symmetric edge index | `id3_ba`, `id3_ca`, `id3_ragged_idx` |

Frozen parameters are cutoff `7.0 Å`, maximum neighbors `50`, and maximum
periodic repetitions per dimension `5`.

## Candidate order and cutoff

For B1 with `n` atoms, linear `k=0..n²-1` produces `index1=floor(k/n)` and
`index2=k mod n`. `index1` is the slow target, `index2` the fast source, and
the returned edge is `(index2,index1)`. Periodic repetitions are recomputed
from reciprocal plane distances. `torch.cartesian_prod` is lexicographic with
the third coordinate fastest. Expansion is pair-major then offset-major.
Candidates survive when FP32 squared distance is `<= cutoff²` and `>0.0001`;
`masked_select` preserves traversal order.

## Top-50 and ties

Candidates are grouped by sorted `index1`. If any target exceeds 50 neighbors,
the reference builds `[num_atoms,max_neighbor_count]`, fills unused slots with
infinity, and calls `torch.sort(..., dim=1)` without `stable=True`. Selected
indices become a mask over the original candidate vector, so output remains in
candidate order. Equal-distance order on the frozen CUDA stack is deterministic
for a fixed shape but unstable. For an exact tie crossing the top-50 boundary,
the MVP compacts the row to the original dynamic width and invokes the same
unstable CUDA sort. This preserves the original tie selection without a
fallback. When the boundary distance is unique, a stable fixed-width sort is
safe because only the selected set matters and output edges are repacked in
the original candidate order.

## Symmetric edges and triplets

Symmetrization retains `source < target`, plus periodic self-atom edges whose
integer offset is lexicographically earlier than zero. It appends reverse
edges, negating their offsets/vectors while copying distance.

Triplets use `SparseTensor(row=target,col=source,value=edge_index)`. For every
output `c->a`, all incoming `b->a` edges are selected. Only identical edge
indices are removed; distinct periodic edges with `b==c` remain. `id3_ca` is
output-edge sorted, `id3_ba` follows sparse storage order, and ragged indices
restart at zero for each output edge.

## Dtypes, allocation, and synchronization

Geometry and distances are FP32 CUDA; atom/edge/triplet indices are long CUDA;
masks are Boolean CUDA. Dynamic work includes `repeat_interleave`, `arange`,
`cartesian_prod`, `repeat`, `stack`, `cat`, `masked_select`, distance-sort
allocation, and per-call `SparseTensor`. Synchronization occurs in PBC
`.item()` checks, Python `int` conversion of repetition maxima, CUDA-tensor
branches, and dynamic shape-producing reductions.

These ordered tensors define correctness; score closeness cannot replace exact
edge, offset, and triplet equality.
