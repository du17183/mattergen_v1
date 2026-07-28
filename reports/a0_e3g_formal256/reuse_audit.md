# A0 formal 256 reuse audit

- Candidate seeds: `20000–20255`
- Physically complete A0 records: `256/256`
- Eligible independent records: `192/256`
- Ineligible records: `64/256`
- Q3 gate-training overlap: `64/256` (`20000–20063`)
- Partial-old plus newly generated mixing is forbidden.
- Terminal state: `SOURCE_DATA_INCOMPLETE`

The A0 files are physically complete, but the registered formal batch is not scientifically independent because 64 seeds trained the frozen Q3 gate. No MatterGen, refinement, or MatterSim task was started.
