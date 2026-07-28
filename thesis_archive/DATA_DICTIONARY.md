# 数据字典

所有 CSV 为 UTF-8，seed 保留真实原值。

| 字段/后缀 | 定义 | 单位/类型 |
|---|---|---|
| `*_max_force` | 松弛前 MatterSim-5M 最大原子力 | eV/Å |
| `*_rmsd` | 松弛前后位置 RMSD | Å |
| `*_ehull` | 代理 E-hull | eV/atom |
| `*_stable`/`*_nus` | 稳定/NUS | bool |
| `force_difference` | 选择方法减基线，负为改善 | eV/Å |
| `gate_confidence`/`gate_on` | Gate 概率/决策 | [0,1]/bool |
| `displacement_max` | wrapped 最大位移 | Å |
| `displacement_mean` | 源数据无逐 seed 值，保留 NaN | Å/缺失 |
| `refinement_harm` | 力差 > 1e-6 | bool |
| `valid_for_formal_claims` | 正式独立资格 | bool |
