from common.archive_utils import DATASETS,load
for name,path in DATASETS.items():
 df=load(name)
 print(f"{name}: rows={len(df)}, columns={len(df.columns)}, seeds={df.seed.min()}..{df.seed.max()}, path={path.relative_to(path.parents[2])}")
