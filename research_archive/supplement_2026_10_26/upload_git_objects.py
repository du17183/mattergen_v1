#!/usr/bin/env python3
"""Publish exact existing Git commits via the Git Data API, without force.

Used when the source machine's Git HTTPS transport fails. Trees and commits
must retain their local SHA; an existing remote ref must be an ancestor.
GitHub CLI supplies credentials internally; this script never reads a token.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def actor(value: str) -> dict:
    match = re.fullmatch(r"(.*) <([^>]*)> (\d+) ([+-])(\d{2})(\d{2})", value)
    if not match:
        raise RuntimeError("Unsupported commit actor header")
    name, email, seconds, sign, hour, minute = match.groups()
    offset = timedelta(hours=int(hour), minutes=int(minute))
    if sign == "-":
        offset = -offset
    return {"name": name, "email": email, "date": datetime.fromtimestamp(int(seconds), timezone(offset)).isoformat(timespec="seconds")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gh", required=True)
    parser.add_argument("--branch", action="append", required=True)
    args = parser.parse_args()

    def api(path: str, payload: dict | None = None, method: str | None = None):
        command = [args.gh, "api", "-H", "Accept: application/vnd.github+json"]
        if payload is not None:
            command += ["-X", method or "POST", "--input", "-"]
        command.append("repos/du17183/mattergen_v1/" + path)
        error = ""
        for attempt in range(5):
            try:
                result = subprocess.run(command, input=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=100)
                if result.returncode:
                    error = result.stderr.decode(errors="replace")[-1200:]
                else:
                    return json.loads(result.stdout)
            except subprocess.TimeoutExpired:
                error = "request timeout"
            time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"GitHub API failed for {path}: {error}")

    refs = api("git/matching-refs/heads/")
    remote = {r["ref"].removeprefix("refs/heads/"):r["object"]["sha"] for r in refs}
    targets = {b:git("rev-parse", b).decode().strip() for b in args.branch}
    for branch, target in targets.items():
        old = remote.get(branch)
        if old and old != target:
            check = subprocess.run(["git", "merge-base", "--is-ancestor", old, target], cwd=REPO)
            if check.returncode:
                raise RuntimeError(f"Remote {branch} is not an ancestor; refusing update")
    commits = git("rev-list", "--reverse", *targets.values(), "--not", "--remotes=origin").decode().splitlines()
    for commit in commits:
        raw = git("cat-file", "-p", commit).decode()
        headers, message = raw.split("\n\n", 1)
        fields = {}; parents = []
        for line in headers.splitlines():
            key, value = line.split(" ", 1)
            if key == "parent": parents.append(value)
            else: fields[key] = value
        if len(parents) != 1:
            raise RuntimeError("Only ordinary single-parent archive commits are supported")
        parent = parents[0]
        base = git("rev-parse", parent + "^{tree}").decode().strip()
        all_entries = {}
        for record in git("ls-tree", "-r", "-z", commit).split(b"\0"):
            if record:
                header, path = record.split(b"\t", 1)
                mode, kind, sha = header.decode().split()
                all_entries[path.decode()] = (mode, kind, sha)
        changed = [r.decode() for r in git("diff", "--name-only", "-z", parent, commit).split(b"\0") if r]
        batches, batch, size = [], [], 0
        for path in changed:
            if path not in all_entries:
                raise RuntimeError("Archive uploader refuses a deletion")
            mode, kind, sha = all_entries[path]
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise RuntimeError("Unsupported tree entry")
            data = git("cat-file", "blob", sha)
            entry = {"path": path, "mode": mode, "type": "blob"}
            try:
                content = data.decode("utf-8")
                if "\0" in content or len(data) > 2 * 1024 * 1024:
                    raise ValueError("Upload this blob separately")
                entry["content"] = content
            except (UnicodeDecodeError, ValueError):
                blob = api("git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
                if blob["sha"] != sha: raise RuntimeError("Blob SHA mismatch")
                entry["sha"] = sha
            length = len(json.dumps(entry, ensure_ascii=False).encode())
            if batch and size + length > 2 * 1024 * 1024:
                batches.append(batch); batch = []; size = 0
            batch.append(entry); size += length
        if batch: batches.append(batch)
        print(f"COMMIT {commit[:12]}: {len(changed)} files / {len(batches)} tree batches", flush=True)
        for number, batch in enumerate(batches, 1):
            base = api("git/trees", {"base_tree": base, "tree": batch})["sha"]
            print(f"TREE_BATCH {number}/{len(batches)}", flush=True)
        if base != fields["tree"]:
            raise RuntimeError("Final tree SHA mismatch; no branch updated")
        created = api("git/commits", {"tree": base, "parents": parents, "message": message,
                                      "author": actor(fields["author"]), "committer": actor(fields["committer"])})
        if created["sha"] != commit:
            raise RuntimeError("Commit SHA mismatch; no branch updated")
        print(f"COMMIT_SHA_VERIFIED {commit}", flush=True)
    latest = {r["ref"].removeprefix("refs/heads/"):r["object"]["sha"] for r in api("git/matching-refs/heads/")}
    for branch, target in targets.items():
        if latest.get(branch) != remote.get(branch):
            raise RuntimeError(f"Remote {branch} changed during upload")
        if latest.get(branch) != target:
            if branch in latest:
                api("git/refs/heads/" + branch, {"sha": target, "force": False}, "PATCH")
            else:
                api("git/refs", {"ref": "refs/heads/" + branch, "sha": target})
        verified = api("git/ref/heads/" + branch)["object"]["sha"]
        if verified != target: raise RuntimeError("Remote SHA verification failed")
        git("update-ref", "refs/remotes/origin/" + branch, target)
        print(f"REMOTE_VERIFIED {branch} {target}", flush=True)


if __name__ == "__main__":
    main()
