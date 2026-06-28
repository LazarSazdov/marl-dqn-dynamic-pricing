#!/usr/bin/env python3
"""Download and organize the raw InsideAirbnb files from configs/data.yaml.

Moves any legacy files from data/ into data/raw/<snapshot>/, skips files
that already exist, streams downloads to a .part file and renames on
success, and checks gzip integrity of everything it fetches.

Usage:
    python3 scripts/download_data.py
    python3 scripts/download_data.py --dry-run
    python3 scripts/download_data.py --snapshot 2026-05-24
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_marl.config import load_config
from airbnb_marl.utils.paths import data_dir, raw_snapshot_dir

CHUNK = 1 << 20


def build_url(base_url: str, snapshot: str, name: str, kind: str) -> str:
    return f"{base_url}/{snapshot}/{kind}/{name}"


def human_size(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "unknown size"
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def gzip_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(CHUNK):
                pass
        return True
    except (OSError, EOFError):
        return False


def migrate_legacy(legacy_name: str, target: Path, dry_run: bool) -> bool:
    legacy_path = data_dir() / legacy_name
    if not legacy_path.exists():
        return False
    print(f"  migrate  {legacy_path.name} -> {target.relative_to(data_dir())}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(target))
    return True


def download(url: str, target: Path, dry_run: bool) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "airbnb-marl/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        print(f"  download {url} ({human_size(total)})")
        if dry_run:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        done = 0
        with open(part, "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r    {human_size(done)} / {human_size(total)} ({pct:.0f}%)",
                          end="", flush=True)
        print()
        part.rename(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--snapshot", help="only process this snapshot date")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    failures = []

    for snapshot, files in cfg["files"].items():
        if args.snapshot and snapshot != args.snapshot:
            continue
        print(f"snapshot {snapshot}:")
        for spec in files:
            name, kind = spec["name"], spec["kind"]
            target = raw_snapshot_dir(snapshot) / name
            if target.exists() and target.stat().st_size > 0:
                print(f"  ok       {name} ({human_size(target.stat().st_size)})")
                continue
            if spec.get("legacy") and migrate_legacy(spec["legacy"], target, args.dry_run):
                pass
            else:
                url = build_url(cfg["base_url"], snapshot, name, kind)
                try:
                    download(url, target, args.dry_run)
                except (urllib.error.URLError, OSError) as exc:
                    print(f"  FAILED   {name}: {exc}")
                    failures.append(f"{snapshot}/{name}")
                    continue
            if not args.dry_run and name.endswith(".gz") and not gzip_ok(target):
                print(f"  FAILED   {name}: gzip integrity check failed, removing")
                target.unlink()
                failures.append(f"{snapshot}/{name}")

    if failures:
        print(f"\n{len(failures)} file(s) failed: {', '.join(failures)}")
        return 1
    print("\nAll data files present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
