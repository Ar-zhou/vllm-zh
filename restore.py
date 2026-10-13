#!/usr/bin/env python3
"""Verify and unpack the dataset archives committed as ordinary Git files."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify(dataset: dict) -> None:
    whole = hashlib.sha256()
    total = 0
    for part in dataset["parts"]:
        path = ROOT / "archives" / part["file"]
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
                whole.update(block)
                size += len(block)
                total += len(block)
        if size != part["bytes"] or digest.hexdigest() != part["sha256"]:
            raise RuntimeError(f"Archive shard check failed: {path}")
    if total != dataset["bytes"] or whole.hexdigest() != dataset["sha256"]:
        raise RuntimeError(f"Archive check failed: {dataset['dataset']}")


def extract(dataset: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unzip = subprocess.Popen(["zstd", "-d", "-q", "-c"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert unzip.stdin and unzip.stdout
    untar = subprocess.Popen(["tar", "-xf", "-", "-C", str(output_dir)], stdin=unzip.stdout)
    unzip.stdout.close()
    try:
        for part in dataset["parts"]:
            with (ROOT / "archives" / part["file"]).open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    unzip.stdin.write(block)
    finally:
        unzip.stdin.close()
    unzip_status = unzip.wait()
    untar_status = untar.wait()
    if unzip_status != 0 or untar_status != 0:
        raise RuntimeError(f"Extraction failed: {dataset['dataset']}")


def main() -> None:
    manifest = json.loads((ROOT / "archive_manifest.json").read_text())
    names = [x["dataset"] for x in manifest["datasets"]]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all"] + names, default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for dataset in manifest["datasets"]:
        if args.dataset not in ("all", dataset["dataset"]):
            continue
        print(f"Verifying {dataset['dataset']}...", flush=True)
        verify(dataset)
        if not args.verify_only:
            print(f"Extracting {dataset['dataset']}...", flush=True)
            extract(dataset, args.output_dir)
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
