# Original C0-B1 joint-CFG hotspot profile

- States: `12`
- Warmup/profile calls: `20` / `30`
- Forward CUDA median: `28.315712 ms`
- K1 inclusive share: `18.449%`
- K2 inclusive share: `31.835%`
- K3 inclusive share: `12.994%`

## Top 20 CUDA operators

| Rank | Operator | CUDA total ms | Calls | Mean us | Input shapes | Device memory bytes |
|---:|---|---:|---:|---:|---|---:|
| 1 | `aten::fill_` | 5.840388 | 7385 | 0.791 | `[[], []]` | 0 |
| 2 | `aten::_assert_async` | 4.141047 | 2330 | 1.777 | `[[]]` | -1536 |
| 3 | `aten::_index_put_impl_` | 3.376516 | 18 | 187.584 | `[[18432], [], [952320], [], []]` | -685707264 |
| 4 | `aten::mm` | 2.969239 | 170 | 17.466 | `[[1860, 512], [512, 512]]` | 652247040 |
| 5 | `aten::_index_put_impl_` | 2.936339 | 27 | 108.753 | `[[10240], [], [507904], [], []]` | -556455936 |
| 6 | `aten::_index_put_impl_` | 2.931402 | 18 | 162.856 | `[[16384], [], [800768], [], []]` | -579637248 |
| 7 | `aten::_index_put_impl_` | 2.739315 | 27 | 101.456 | `[[9216], [], [464896], [], []]` | -502142976 |
| 8 | `aten::arange` | 2.655750 | 2145 | 1.238 | `[[], [], [], [0]]` | 0 |
| 9 | `aten::mm` | 2.590898 | 255 | 10.160 | `[[992, 512], [512, 512]]` | 518062080 |
| 10 | `aten::mm` | 2.587771 | 255 | 10.148 | `[[908, 512], [512, 512]]` | 474193920 |
| 11 | `aten::_index_put_impl_` | 2.509355 | 27 | 92.939 | `[[8192], [], [409600], [], []]` | -448321536 |
| 12 | `aten::_index_put_impl_` | 2.394464 | 18 | 133.026 | `[[12288], [], [618496], [], []]` | -445353984 |
| 13 | `aten::_index_put_impl_` | 2.210212 | 18 | 122.790 | `[[12288], [], [557056], [], []]` | -401117184 |
| 14 | `aten::_index_put_impl_` | 2.104354 | 18 | 116.909 | `[[10240], [], [520192], [], []]` | -376934400 |
| 15 | `aten::_index_put_impl_` | 2.087047 | 27 | 77.298 | `[[6144], [], [307200], [], []]` | -332775936 |
| 16 | `aten::mm` | 2.085775 | 469 | 4.447 | `[[20, 512], [512, 512]]` | 19210240 |
| 17 | `aten::_index_put_impl_` | 2.014218 | 18 | 111.901 | `[[10240], [], [512000], [], []]` | -372609024 |
| 18 | `aten::mm` | 2.009075 | 255 | 7.879 | `[[800, 512], [512, 512]]` | 447965184 |
| 19 | `aten::mm` | 1.793551 | 170 | 10.550 | `[[1564, 512], [512, 512]]` | 569999360 |
| 20 | `aten::mm` | 1.786056 | 255 | 7.004 | `[[600, 512], [512, 512]]` | 313344000 |

## Module/source mapping

- `K1_graph_geometry`: 30 calls, 4.925741 ms/call, inputs 311 B, outputs 1096985 B; modules: mattergen/common/gemnet/gemnet.py:generate_interaction_graph
- `K1_radial_basis`: 60 calls, 0.147766 ms/call, inputs 3618 B, outputs 463121 B; modules: cbf_basis3.radial_basis:RadialBasis, radial_basis:RadialBasis
- `K2_atom_update`: 120 calls, 0.715387 ms/call, inputs 1955703 B, outputs 38093 B; modules: int_blocks.0.atom_update:AtomUpdateBlock, int_blocks.1.atom_update:AtomUpdateBlock, int_blocks.2.atom_update:AtomUpdateBlock, int_blocks.3.atom_update:AtomUpdateBlock
- `K2_output_update`: 150 calls, 1.229605 ms/call, inputs 1955703 B, outputs 3693 B; modules: out_blocks.0:OutputBlock, out_blocks.1:OutputBlock, out_blocks.2:OutputBlock, out_blocks.3:OutputBlock, out_blocks.4:OutputBlock
- `K3_lattice_score_head`: 150 calls, 0.735497 ms/call, inputs 0 B, outputs 72 B; modules: lattice_out_blocks.0:RBFBasedLatticeUpdateBlockFrac, lattice_out_blocks.1:RBFBasedLatticeUpdateBlockFrac, lattice_out_blocks.2:RBFBasedLatticeUpdateBlockFrac, lattice_out_blocks.3:RBFBasedLatticeUpdateBlockFrac, lattice_out_blocks.4:RBFBasedLatticeUpdateBlockFrac
- `triplet_interaction_context`: 120 calls, 0.752545 ms/call, inputs 4694303 B, outputs 1852484 B; modules: int_blocks.0.trip_interaction:TripletInteraction, int_blocks.1.trip_interaction:TripletInteraction, int_blocks.2.trip_interaction:TripletInteraction, int_blocks.3.trip_interaction:TripletInteraction

K1/K2/K3 shares use disjoint CUDA-event module boundaries. K2 covers
the repeated dense→gate→scatter→atom/output residual-update family;
triplet interaction is reported separately and is not added to K2.
