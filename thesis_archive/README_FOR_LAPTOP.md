# 在个人笔记本重算统计和图表

只需 Python CPU，不需 MatterGen、MatterSim、PyTorch GPU、CUDA、权重或服务器 Conda。

## Windows PowerShell

```powershell
git clone https://github.com/du17183/mattergen_v1.git
cd mattergen_v1
git switch archive/thesis-analysis-package-v1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r thesis_archive/requirements-analysis.txt
python thesis_archive/analysis/validate_archive.py
python thesis_archive/analysis/recompute_statistics.py
python thesis_archive/analysis/build_result_tables.py
python thesis_archive/analysis/generate_figures.py
```

## Linux/macOS

```bash
git clone https://github.com/du17183/mattergen_v1.git
cd mattergen_v1
git switch archive/thesis-analysis-package-v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r thesis_archive/requirements-analysis.txt
python thesis_archive/analysis/validate_archive.py
python thesis_archive/analysis/recompute_statistics.py
python thesis_archive/analysis/build_result_tables.py
python thesis_archive/analysis/generate_figures.py
```

脚本从自身位置解析仓库相对路径；ZIP 解压后同样可运行。
