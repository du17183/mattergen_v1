#!/usr/bin/env python3
"""Upload verified non-weight bundles into a new, non-latest GitHub Release.

Existing uploaded assets are never overwritten. A draft is published only once
every asset matches its size and SHA256 (server digest or streamed download).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TAG = "mattergen-nonweight-20261026"
TITLE = "MatterGen non-weight research archive — 2026-10-26"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gh", required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    validation = json.loads((HERE / "validation_report.json").read_text())
    if validation["status"] != "PASS":
        raise RuntimeError("Local archive validation must pass first")

    def api(path: str, payload: dict | None = None, method: str | None = None):
        command = [args.gh, "api", "-H", "Accept: application/vnd.github+json"]
        if method or payload is not None:
            command += ["-X", method or "POST"]
        if payload is not None:
            command += ["--input", "-"]
        command += ["repos/du17183/mattergen_v1/" + path]
        error = ""
        for attempt in range(4):
            try:
                result = subprocess.run(command, input=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=100)
                if result.returncode == 0:
                    return json.loads(result.stdout) if result.stdout.strip() else {}
                error = result.stderr.decode(errors="replace")[-1200:]
            except subprocess.TimeoutExpired:
                error = "request timed out"
            time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"GitHub API failed for {path}: {error}")

    config = json.loads((HERE / "assets.json").read_text())
    files = [args.assets_dir / row["name"] for row in config["assets"]]
    for path, row in zip(files, config["assets"]):
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("Local asset differs from the validated manifest")
    kit = args.assets_dir / "mattergen-restore-kit-20261026.tar.gz"
    if not kit.exists():
        with tarfile.open(kit, "w:gz") as tar:
            for name in ["README.md", "restore_snapshot.py", "files.jsonl", "excluded.jsonl", "worktrees.json", "assets.json", "snapshot_summary.json", "validation_report.json", "preflight_scan.json", "private_branch_preflight.json"]:
                tar.add(HERE / name, arcname="restore-kit/" + name, recursive=False)
    files.append(kit)
    sums = args.assets_dir / "SHA256SUMS.txt"
    sums.write_text("".join(sha256(path) + "  " + path.name + "\n" for path in files), encoding="utf-8")
    files.append(sums)
    expected = {p.name:{"sha256":sha256(p), "size_bytes":p.stat().st_size} for p in files}

    releases = api("releases?per_page=100")
    release = next((r for r in releases if r["tag_name"] == TAG), None)
    if release is None:
        notes = (HERE / "release_notes.md").read_text()
        release = api("releases", {"tag_name": TAG, "target_commitish": args.commit, "name": TITLE,
                                    "body": notes, "draft": True, "prerelease": False, "make_latest": "false"})
    elif release["name"] != TITLE or release["target_commitish"] != args.commit:
        raise RuntimeError("Existing release has a different owner/purpose/commit; refusing reuse")
    receipt = []
    for path in files:
        name = path.name
        want = expected[name]
        uploaded = None
        for attempt in range(5):
            state = api(f"releases/{release['id']}")
            asset = next((a for a in state["assets"] if a["name"] == name), None)
            if asset and asset.get("state") == "uploaded":
                if asset["size"] != want["size_bytes"]:
                    raise RuntimeError(f"Existing asset has a different size: {name}")
                uploaded = asset
                break
            if asset:
                # Only an incomplete upload from this exact new release is
                # removed. Successful assets and previous releases are intact.
                if asset.get("state") != "starter":
                    raise RuntimeError(f"Unexpected upload state: {name}")
                api(f"releases/assets/{asset['id']}", method="DELETE")
            print(f"UPLOADING {name} bytes={want['size_bytes']} attempt={attempt+1}", flush=True)
            try:
                result = subprocess.run([args.gh, "release", "upload", TAG, str(path), "--repo", "du17183/mattergen_v1"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
                if result.returncode:
                    print(f"UPLOAD_RETRY {name}: CLI exit {result.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"UPLOAD_RETRY {name}: timeout", flush=True)
            time.sleep(2)
        if uploaded is None:
            state = api(f"releases/{release['id']}")
            uploaded = next((a for a in state["assets"] if a["name"] == name and a.get("state") == "uploaded"), None)
        if uploaded is None or uploaded["size"] != want["size_bytes"]:
            raise RuntimeError(f"Upload did not complete: {name}")
        digest = uploaded.get("digest")
        if digest:
            if digest != "sha256:" + want["sha256"]:
                raise RuntimeError(f"Server SHA256 mismatch: {name}")
            verification = "github_server_sha256"
        else:
            process = subprocess.Popen([args.gh, "api", f"repos/du17183/mattergen_v1/releases/assets/{uploaded['id']}",
                                        "-H", "Accept: application/octet-stream"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            h = hashlib.sha256(); total = 0
            for block in iter(lambda: process.stdout.read(1024*1024), b""):
                h.update(block); total += len(block)
            if process.wait() != 0 or h.hexdigest() != want["sha256"] or total != want["size_bytes"]:
                raise RuntimeError(f"Downloaded asset verification failed: {name}")
            verification = "streamed_download_sha256"
        receipt.append({"name": name, **want, "github_asset_id": uploaded["id"], "url": uploaded["browser_download_url"], "verification": verification})
        print(f"REMOTE_ASSET_VERIFIED {name} sha256={want['sha256']}", flush=True)
    published = api(f"releases/{release['id']}", {"draft": False, "make_latest": "false"}, "PATCH")
    if published["draft"]:
        raise RuntimeError("Release remained a draft")
    report = {"status": "PASS", "source_commit": args.commit, "tag": TAG, "url": published["html_url"], "assets": receipt}
    (HERE / "upload_receipt.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RELEASE_PUBLISHED {published['html_url']} assets={len(receipt)}", flush=True)


if __name__ == "__main__":
    main()
