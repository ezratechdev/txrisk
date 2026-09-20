"""Download benchmark datasets into data/raw/<name>/ and verify their checksums.

    python -m txrisk.data.download elliptic [bitcoinheist ...]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

from txrisk.data.registry import DATASETS, Dataset
from txrisk.paths import RAW_DIR

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Stream url to dest via a .part file, so an interrupted download never looks complete."""
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "txrisk"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        for block in iter(lambda: response.read(CHUNK), b""):
            out.write(block)
            done += len(block)
            share = f"{done / total:6.1%} of " if total else ""
            progress = f"{share}{(total or done) / 1e6:.0f} MB"
            print(f"\r  {dest.name}: {progress}", end="", file=sys.stderr)
    print(file=sys.stderr)
    part.replace(dest)


def extract(archive: Path, out: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        if all(
            (out / m.filename).exists() and (out / m.filename).stat().st_size == m.file_size
            for m in members
        ):
            return
        z.extractall(out)


def download(dataset: Dataset, root: Path = RAW_DIR, force: bool = False) -> Path:
    out = root / dataset.name
    out.mkdir(parents=True, exist_ok=True)
    for remote in dataset.files:
        archive = out / remote.url.rsplit("/", 1)[-1]
        if force or not archive.exists():
            fetch(remote.url, archive)
        digest = sha256_of(archive)
        if remote.sha256 is None:
            print(f"  {archive.name}: sha256 {digest} (not pinned in registry.py yet)")
        elif digest != remote.sha256:
            raise RuntimeError(
                f"{archive.name}: checksum mismatch (expected {remote.sha256}, got {digest}). "
                "Delete the file and download again; if it persists, the source has changed."
            )
        if zipfile.is_zipfile(archive):
            extract(archive, out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument("names", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args(argv)
    for name in args.names:
        dataset = DATASETS[name]
        print(f"{name}: {dataset.description}\n  source: {dataset.source}")
        print(f"  saved to {download(dataset, force=args.force)}")


if __name__ == "__main__":
    main()
