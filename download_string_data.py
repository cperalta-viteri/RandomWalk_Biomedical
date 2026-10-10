#!/usr/bin/env python3
"""Download and checksum-verify the versioned STRING inputs used by this tutorial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "provenance" / "string_v12_5_source_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(item: dict, destination: Path, overwrite: bool) -> None:
    target = destination / item["file"]
    if target.exists():
        observed = sha256(target)
        if observed == item["sha256"]:
            print(f"verified: {target.name}")
            return
        if not overwrite:
            raise RuntimeError(
                f"{target.name} exists but its checksum differs. "
                "Use --overwrite only if you intend to replace it."
            )
    temporary = target.with_name(f"{target.name}.part")
    if temporary.exists():
        temporary.unlink()
    print(f"downloading: {item['url']}")
    try:
        with urllib.request.urlopen(item["url"]) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        observed = sha256(temporary)
        if observed != item["sha256"]:
            raise RuntimeError(f"checksum mismatch after download for {target.name}")
        os.replace(temporary, target)
        print(f"verified: {target.name}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="directory for official STRING archives (default: RWR/data)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing archive only when its checksum differs")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        download(item, args.data_dir, args.overwrite)
    print("STRING data are ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"download failed: {error}", file=sys.stderr)
        raise SystemExit(1)

