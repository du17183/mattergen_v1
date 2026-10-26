#!/usr/bin/env python3
"""Additive archive of uncommitted research files; never runs an experiment.

Source worktrees are read-only. Browseable source/report copies are accompanied
by content-addressed Release bundles containing every included overlay file.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TAG = "mattergen-nonweight-20261026"
URL = "https://github.com/du17183/mattergen_v1/releases/download/" + TAG
SCOPES = {"experiments", "results", "logs", "docs", "scripts", "figures", "mattergen", "tests", "configs", "reports"}
CACHE_DIRS = {".git", ".venv", ".envs", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache", "runtime_tmp", "tmp", "temp", "node_modules", ".ipynb_checkpoints"}
TEXT_EXT = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".txt", ".json", ".jsonl", ".csv", ".log", ".out", ".err", ".tex", ".xml", ".svg", ".patch", ".ini", ".cfg"}
RAW_DIRS = {"generation", "structures", "relaxation", "diagnostic_runs", "diagnostic_states", "quality", "runtime", "teacher_data", "dagger_data"}
PATTERNS = {
    "github_token": re.compile(rb"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    "api_key": re.compile(rb"sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_]{24,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "authenticated_url": re.compile(rb"https?://[^\s/'\"<>:@]+:[^\s/'\"<>@]{4,}@"),
}


def git(*args: str, cwd: Path = REPO) -> bytes:
    return subprocess.check_output(["git", *args], cwd=cwd)


def dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_policy(relative: str) -> str | None:
    p = Path(relative)
    lower = relative.lower()
    if any(part in CACHE_DIRS for part in p.parts):
        return "runtime_cache_or_environment"
    if p.name.startswith(".env") or p.name in {"hosts.yml", ".netrc", "credentials", "id_rsa", "id_ed25519"}:
        return "private_configuration"
    if p.suffix in {".pyc", ".pyo", ".aux", ".toc", ".lof", ".lot", ".synctex"}:
        return "bytecode_or_build_cache"
    if lower.endswith((".pth.tar", ".safetensors", ".ckpt", ".pth", ".onnx")):
        return "model_weights"
    if p.suffix == ".pt":
        if p.name == "prefix_checkpoint.pt" or "diagnostic_states" in p.parts or "teacher_data" in p.parts or "dagger_data" in p.parts or p.name == "stage3a_inference_meta.pt":
            return None  # Diffusion/RNG states or generated training examples, not weights.
        return "model_weights_or_unclassified_torch_checkpoint"
    if p.suffix in {".lmdb", ".db", ".mdb"} or "reference_mp2020correction" in lower or "reference_tri2024correction" in lower:
        return "external_reference_database"
    if p.suffix == ".bin":
        return "unclassified_model_binary"
    return None


def scan_bytes(data: bytes, path: str) -> list[dict]:
    return [{"path": path, "pattern": name} for name, pattern in PATTERNS.items() if pattern.search(data)]


def scan_text(path: Path, logical: str) -> list[dict]:
    if path.suffix.lower() not in TEXT_EXT:
        return []
    findings = []
    with path.open("rb") as stream:
        tail = b""
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            findings.extend(scan_bytes(tail + block, logical))
            tail = block[-512:]
    return findings


def browseable(relative: str, size: int) -> bool:
    p = Path(relative)
    if size > 2 * 1024 * 1024:
        return False
    if p.suffix in {".py", ".sh", ".yaml", ".yml", ".toml", ".md"}:
        return True
    if any(part in RAW_DIRS or part.startswith("p0_relaxation") or part.startswith("p0_quality") for part in p.parts):
        return False
    return p.suffix in {".json", ".csv", ".txt", ".jsonl"} and size <= 512 * 1024


def inventory() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    records, excluded, worktrees, findings = [], [], [], []
    roots = [Path(line[9:]) for line in git("worktree", "list", "--porcelain").decode().splitlines() if line.startswith("worktree ")]
    for root in roots:
        # The archive is the destination. Its existing experiments are clean
        # tracked source, so skipping it also prevents self-inclusion.
        if root == REPO:
            continue
        name = root.name
        head = git("rev-parse", "HEAD", cwd=root).decode().strip()
        branch = git("branch", "--show-current", cwd=root).decode().strip()
        tracked = {p.decode() for p in git("ls-files", "-z", cwd=root).split(b"\0") if p}
        changed = {p.decode() for p in git("diff", "HEAD", "--name-only", "-z", cwd=root).split(b"\0") if p}
        status = git("status", "--porcelain=v1", "--untracked-files=normal", cwd=root).decode()
        worktrees.append({"name": name, "original_root": str(root), "branch": branch, "commit": head,
                          "tracked_changes": sorted(changed), "status_before": status})
        candidates = set(changed)
        for scope in sorted(SCOPES):
            start = root / scope
            if not start.is_dir():
                continue
            for d, dirs, files in os.walk(start, followlinks=False):
                for sub in list(dirs):
                    q = Path(d) / sub
                    if sub in CACHE_DIRS or q.is_symlink():
                        dirs.remove(sub)
                        excluded.append({"worktree": name, "path": q.relative_to(root).as_posix(), "kind": "directory", "reason": "runtime_cache_environment_or_directory_symlink"})
                for filename in files:
                    relative = (Path(d) / filename).relative_to(root).as_posix()
                    if relative not in tracked or relative in changed:
                        candidates.add(relative)
        for relative in sorted(candidates):
            p = root / relative
            reason = file_policy(relative)
            if reason:
                size = p.stat().st_size if p.is_file() else 0
                row = {"worktree": name, "path": relative, "size_bytes": size, "reason": reason}
                if reason.startswith("model_weights") and p.is_file():
                    row["sha256"] = sha256(p)
                excluded.append(row)
                continue
            if not p.exists():
                records.append({"worktree": name, "path": relative, "kind": "deleted", "base_commit": head})
                continue
            if not p.is_file():
                excluded.append({"worktree": name, "path": relative, "reason": "not_a_regular_file"})
                continue
            link_target = None
            if p.is_symlink():
                target = p.resolve()
                if not any(target.is_relative_to(r) for r in roots):
                    excluded.append({"worktree": name, "path": relative, "reason": "symlink_outside_project"})
                    continue
                link_target = os.readlink(p)
            info = p.stat()
            digest = sha256(p)
            findings.extend(scan_text(p, f"{name}/{relative}"))
            if p.suffix == ".zip":
                with zipfile.ZipFile(p) as z:
                    for member in z.infolist():
                        if member.is_dir():
                            continue
                        why = file_policy(member.filename)
                        if why and why.startswith("model_weights"):
                            findings.append({"path": f"{name}/{relative}", "pattern": "nested_model_weight"})
                        if Path(member.filename).suffix in TEXT_EXT and member.file_size < 32 * 1024 * 1024:
                            findings.extend(scan_bytes(z.read(member), f"{name}/{relative}!{member.filename}"))
            records.append({"worktree": name, "path": relative, "kind": "file", "base_commit": head,
                            "size_bytes": info.st_size, "sha256": digest, "mode": stat.S_IMODE(info.st_mode),
                            "mtime_ns": info.st_mtime_ns, "source": str(p), "source_symlink_target": link_target,
                            "browseable": browseable(relative, info.st_size)})
        print(f"INVENTORY {name}: {sum(r['worktree']==name for r in records)} overlay files", flush=True)
    return records, excluded, worktrees, findings


def pack(records: list[dict], assets_dir: Path, limit: int) -> list[dict]:
    unique = {}
    for row in records:
        if row["kind"] == "file":
            unique.setdefault(row["sha256"], row)
    groups, group, size = [], [], 0
    for digest, row in sorted(unique.items()):
        if group and size + row["size_bytes"] > limit:
            groups.append(group); group = []; size = 0
        group.append(row); size += row["size_bytes"]
    if group:
        groups.append(group)
    assets, locations = [], {}
    for number, items in enumerate(groups, 1):
        name = f"mattergen-nonweight-20261026-part{number:02d}.tar.zst"
        target = assets_dir / name
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite existing asset: {target}")
        partial = target.with_suffix(target.suffix + ".partial")
        proc = subprocess.Popen(["zstd", "-T2", "-6", "-q", "-o", str(partial)], stdin=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                for row in items:
                    p = Path(row["source"]); before = p.stat()
                    if (before.st_size, before.st_mtime_ns) != (row["size_bytes"], row["mtime_ns"]):
                        raise RuntimeError(f"Source changed while archiving: {p}")
                    info = tarfile.TarInfo("blobs/" + row["sha256"])
                    info.size = row["size_bytes"]; info.mode = 0o644; info.mtime = 0
                    with p.open("rb") as stream:
                        tar.addfile(info, stream)
                    locations[row["sha256"]] = name
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("zstd failed")
        except BaseException:
            proc.kill(); proc.wait(); raise
        partial.rename(target)
        assets.append({"name": name, "sha256": sha256(target), "size_bytes": target.stat().st_size,
                       "blob_count": len(items), "uncompressed_bytes": sum(r["size_bytes"] for r in items),
                       "download_url": URL + "/" + name})
        print(f"PACKED {name}: {target.stat().st_size} bytes; {len(items)} verified-source blobs", flush=True)
    for row in records:
        if row["kind"] == "file":
            row["asset"] = locations[row["sha256"]]
    return assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--part-mib", type=int, default=300)
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    records, excluded, worktrees, findings = inventory()
    dump(OUT / "preflight_scan.json", {"status": "FAIL" if findings else "PASS", "findings_without_values": findings})
    if findings:
        raise RuntimeError(f"Credential/container preflight failed ({len(findings)} findings); see path-only report")
    assets = pack(records, args.assets_dir, args.part_mib * 1024 * 1024)
    for row in records:
        if row["kind"] == "file" and row["browseable"]:
            copy = OUT / "source_snapshots" / row["worktree"] / row["path"]
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["source"], copy)
            os.chmod(copy, row["mode"] & 0o777)
            if sha256(copy) != row["sha256"]:
                raise RuntimeError(f"Browseable copy SHA mismatch: {copy}")
            row["git_copy"] = copy.relative_to(REPO).as_posix()
    for tree in worktrees:
        path = OUT / "worktree_status" / (tree["name"] + ".patch")
        patch = git("diff", "HEAD", "--binary", cwd=Path(tree["original_root"]))
        if patch:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(patch)
    manifest = OUT / "files.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps({k:v for k,v in row.items() if k not in {"mtime_ns", "browseable"}}, ensure_ascii=False, sort_keys=True) + "\n")
    with (OUT / "excluded.jsonl").open("w", encoding="utf-8") as stream:
        for row in excluded:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    dump(OUT / "worktrees.json", worktrees)
    dump(OUT / "assets.json", {"release_tag": TAG, "assets": assets})
    summary = {"date": "2026-10-26", "source_worktrees": len(worktrees), "overlay_files": len(records),
               "included_bytes_before_dedup": sum(r.get("size_bytes",0) for r in records),
               "browseable_copies": sum("git_copy" in r for r in records),
               "excluded_by_reason": dict(collections.Counter(r["reason"] for r in excluded)),
               "release_assets": len(assets), "release_asset_bytes": sum(a["size_bytes"] for a in assets),
               "manifest_sha256": sha256(manifest), "source_files_deleted": 0,
               "note": "Complete overlay on each recorded Git commit, with weights/caches/private configuration excluded. Raw tensors are included only for named generated datasets or diffusion/RNG states."}
    dump(OUT / "snapshot_summary.json", summary)
    checksum = args.assets_dir / "SHA256SUMS.txt"
    checksum.write_text("".join(a["sha256"] + "  " + a["name"] + "\n" for a in assets), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
