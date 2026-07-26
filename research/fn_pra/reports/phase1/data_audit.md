# FN-PRA Phase-1 data audit

Generated: `2026-07-25T20:17:10+08:00`

- Official archive: `/data/dxl/data/archives/mp_20.zip`
- Archive SHA256: `5f44d6ad75299b6c08a3679ebcbd166def45c6d7733baacdce3972c2ec452e3a`
- Cache: `/data/dxl/datasets/cache/mp_20`
- Probe subset: 1000 train structures, seed 20260725
- Mapping: strict `(structure_id, atom_index)` mapping is established.
- Default ChemGraph does not expose structure_id; FN-PRA uses a cache-backed dataset that attaches atom rows before collate.

| split | structures | atoms | dft_mag non-null |
|---|---:|---:|---:|
| train | 27136 | 279553 | 26117 |
| val | 9047 | 92454 | 8694 |
| test | 9046 | 92085 | 8716 |
