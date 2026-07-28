# 未上传 GitHub 的工件

目录 manifest 哈希口径：对目录内全部文件的相对路径与字节大小按字典序排序后计算 SHA256；它用于识别目录快照，不等价于逐文件内容哈希。所有这些工件均不是笔记本统计和绘图所必需。

| 工件 | 原服务器路径 | 用途 | 大小与哈希 | 未上传原因 | 可重新获得/生成 |
|---|---|---|---|---|---|
| MatterGen 条件权重 | `/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt` | C0/A0 生成 | 511,777,278 bytes；SHA256 `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e` | 大型权重/用户明确排除 | 官方 MatterGen 来源 |
| MatterSim-5M | `/data/dxl/mattersim_weights/mattersim-v1.0.0-5M.pth` | 代理松弛与评价 | 91,176,875 bytes；SHA256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5` | 权重/用户明确排除 | MatterSim 官方来源 |
| Q3 Gate checkpoint | `/data/dxl/results/postgen_fastgate/q3_refiner/model/q3_gate.joblib` | Learned Gate，129 参数 | 34,321 bytes；SHA256 `b2ce1800fa0fa448f57d58010c8586b5de1b6666c4f198737a2f8a4bfabcb90e` | 属于权重，按规则排除 | 正式训练产物；训练 seeds 20000–20063；加载逻辑 `research/postgen_fastgate/refiner_eval.py`；配置 `configs/e3_pcr_final.yaml`，配置 SHA256 `50d10efdea1050a84de6b2872f78742c2468ff4bef45cd7544fb30cef31eb87a` |
| Conda 环境 | `/data/dxl/envs/mattergen_py310` | 原实验运行 | 7,718,164,182 bytes；48,693 files；path-size manifest SHA256 `605207b971211980f267ed0bef7b32ef5998ef0ba6dd3f28a97f433a9cd5cfdb` | 环境体积大 | 用环境规范重建；笔记本分析只需 `requirements-analysis.txt` |
| 数据与小型/大型缓存全集 | `/data/dxl/data` | 训练数据、映射与缓存 | 316,631,812 bytes；1,650 files；path-size manifest SHA256 `a1021383bd767f1cd351fcf036b374810789b80ca84fb442d3c858cc902bf6e1` | 原始数据集/cache 不上传 | 按各正式流程来源重建 |
| 结果与结构轨迹全集 | `/data/dxl/results` | 生成、精修、松弛中间物 | 8,618,957,415 bytes；65,201 files；path-size manifest SHA256 `fed73107fe8dca8b8735b49107d61684483136b8ba74ee656a555691089fad2d` | 大型结构缓存/轨迹 | 由各冻结运行命令重建；论文分析使用已归档逐 seed 指标 |
| 日志全集 | `/data/dxl/logs` | 运行诊断 | 977,721,875 bytes；3,745 files；path-size manifest SHA256 `2128a0ba865c2757028b8ecb14ce5ceeee6b5f598d762d29dc7957b6b245e70f` | 大型日志 | 可由运行过程重新产生 |

服务器绝对路径只出现在本审计清单和各 `source_manifest.json` 的来源字段；便携分析代码与配置不依赖这些路径。未上传任何密钥、Token、Hugging Face cache 或 Python package cache。
