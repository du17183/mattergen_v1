# 本轮测试记录

2026-09-14，使用 /mnt/datasets-livsyn/dxl/alm/.venv/bin/python。

1. Adaptive CFG 原有控制器测试：13 passed（test_guidance_schedule.py）。
2. 旧 finalization 的 test_analysis.py + test_mechanisms.py，以及本轮 test_final_statistics.py：14 passed。
3. 独立证据核验：verify_evidence.py 对历史方向事件、范数、POST 原 C0 哈希和全部真实调用计数完成复核。新增最小消融的完整轨迹核验必须等待其实际完成，以 evidence_verification.json 为准。

测试工程问题保留：第一次运行旧机制测试时 PYTHONPATH 未包含 closed_loop_guidance worktree，collection 阶段报 ModuleNotFoundError，未执行/修改任何实验。将现有 worktree 加入测试 import path 后，14 项测试全部通过。不涉及改变 F1 方法或重启其研究路线。

最终审计必须另行验证章节节数、Q1–Q17、图像完整性、链接存在、历史 NA 字段、原始 bootstrap 复算以及所有来源哈希；测试通过本身不代表全部论文实验已经完成。
