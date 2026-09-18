# Large-file audit

Audit date: 2026-09-18

Status: `LARGE_FILE_AUDIT = PASS`

## Threshold summary

| Scope | >10 MB | >50 MB | >100 MB | Result |
|---|---:|---:|---:|---|
| New `thesis_release/` package | 0 | 0 | 0 | PASS |
| Full checked-out repository | 1 | 1 | 1 | Existing Git LFS asset; no new ordinary Git blob |

The largest file added under `thesis_release/` is
`innovation1/figures/I1_F2_fixed_vs_c0_paired.png` at 354,473 bytes. No model
weights, checkpoints, LMDB files, raw structure batches, caches, environments or
trajectory dumps were added.

## Existing upstream LFS asset

`data-release/alex-mp/reference_MP2020correction.gz` is 873,410,170 bytes in the
smudged working tree. It predates this release and has Git attributes
`filter=lfs`, `diff=lfs`, `merge=lfs`, `-text`. The object committed in Git is a
134-byte LFS pointer with declared SHA-256 object ID
`c722f72c7d0cd398fc382870f0b731425b54bd4492b089ca36e3119e49f469b5`.
It was neither added nor modified by the thesis release.

## Commands used

```bash
find . -type f -printf '%s %p\n' | sort -nr | head -100
find . -type f -size +10M -printf '%s %p\n' | sort -nr
find thesis_release -type f -size +10M -printf '%s %p\n' | sort -nr
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr
git check-attr -a -- data-release/alex-mp/reference_MP2020correction.gz
git cat-file -s HEAD:data-release/alex-mp/reference_MP2020correction.gz
```

The release satisfies the policy of no new file above 50 MB and no ordinary Git
blob above 100 MB.
