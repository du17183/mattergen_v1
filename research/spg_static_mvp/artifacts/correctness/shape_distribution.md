# SPG static MVP real trajectory shape distribution

- Real states: 64000
- Independent C0 samples: 32
- Selected bucket: `atoms_8_12_rep_le_2`
- State coverage: 33.6578%
- Sample coverage: 62.5000%
- Candidate-capacity padding waste: 37.5902%
- Atom padding waste: 15.7750%
- Candidate pair-image capacity: 18000
- Edge capacity: 626
- Triplet capacity: 32156

## Quantiles

| metric                    |   count |         p50 |         p75 |         p90 |         p95 |         p99 |        max |
|:--------------------------|--------:|------------:|------------:|------------:|------------:|------------:|-----------:|
| num_atoms                 |   64000 |    10       |    12       |    15       |    16       |     18      |     18     |
| candidate_periodic_images |   64000 |   125       |   225       |   405       |   693       |   1331      |   1331     |
| candidate_pair_images     |   64000 | 12500       | 24500       | 40500       | 68607       | 161051      | 431244     |
| raw_edge_count            |   64000 |   500       |   600       |   750       |   800       |    900      |    900     |
| edge_count                |   64000 |   500       |   594       |   754       |   804       |    902      |    930     |
| max_neighbors             |   64000 |    52       |    53       |    55       |    56       |     58      |     69     |
| mean_neighbors            |   64000 |    50       |    50.3636  |    50.6667  |    50.9091  |     51.3333 |     55     |
| triplet_count             |   64000 | 24348       | 28822       | 37204       | 39622.1     |  44342      |  47250     |
| cell_volume               |   64000 |   128.754   |   189.404   |   266.15    |   326.362   |    444.544  |    788.726 |
| cell_condition_number     |   64000 |     2.67403 |     3.75182 |     5.72443 |     8.17844 |     25.7034 |  16768.7   |

## Atom-count coverage

|   num_atoms |   states |   samples |   state_coverage |   sample_coverage |
|------------:|---------:|----------:|-----------------:|------------------:|
|           4 |     8000 |         4 |          0.125   |           0.125   |
|           5 |     2000 |         1 |          0.03125 |           0.03125 |
|           6 |     4000 |         2 |          0.0625  |           0.0625  |
|           8 |     6000 |         3 |          0.09375 |           0.09375 |
|           9 |     4000 |         2 |          0.0625  |           0.0625  |
|          10 |    18000 |         9 |          0.28125 |           0.28125 |
|          11 |     4000 |         2 |          0.0625  |           0.0625  |
|          12 |     8000 |         4 |          0.125   |           0.125   |
|          14 |     2000 |         1 |          0.03125 |           0.03125 |
|          15 |     2000 |         1 |          0.03125 |           0.03125 |
|          16 |     4000 |         2 |          0.0625  |           0.0625  |
|          18 |     2000 |         1 |          0.03125 |           0.03125 |

The atom-count and periodic-repetition bucket is frozen before builder implementation.
