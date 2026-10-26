# DFT 环境审计

DFT_VERIFICATION=NOT_RUN_ENVIRONMENT_UNAVAILABLE

实际检查 PATH、dxl 项目目录、/opt、/usr/local/bin、/usr/bin。未找到 VASP、Quantum ESPRESSO、ABINIT 或 CASTEP 可执行程序。检出文件仅为 ASE fake_pseudos 测试 POTCAR，不可用于真实计算。未验证许可授权；不能因没找到许可文件断言用户无许可。未找到已验证计算工作流或调度器；CPU/RAM 充足本身不等于具备 DFT 运行条件。

完成的是环境审计，不是 DFT 计算尝试。实际完成单点数=0；不生成虚构的 16-pair 选择/结果。可选 DFT 不阻止收尾，DFT_VERIFIED=False。

详见 environment_audit.json 的原始二进制和文件检查列表。
