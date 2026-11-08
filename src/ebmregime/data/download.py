"""Verified, resumable download of archive files, plus the data manifest.

A file is kept only if its SHA-256 matches the archive's ``.CHECKSUM`` sidecar. Files are written
atomically (temporary file, then rename), so an interrupted run never leaves a partial zip under its
final name. A file already on disk whose checksum still verifies is not downloaded again.

The manifest (``data/manifest.csv``) records every file with its checksum, size, the archive's
Last-Modified time and our download time. Binance occasionally replaces archive files, and
``manifest_hash`` feeds the cache keys of later stages.
"""

from __future__ import annotations

import csv
import hashlib
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

USER_AGENT = "ebmregime-research-fetch/0.1 (+non-commercial research)"
MANIFEST_FIELDS = ["symbol", "interval", "file", "url", "sha256", "size_bytes",
                   "last_modified", "downloaded_at"]


class ChecksumError(RuntimeError):
    pass


def make_fetcher(timeout_s: float = 60, retries: int = 4, backoff_s: float = 2.0) -> Callable[[str], bytes]:
    """HTTP GET with retries and exponential backoff on network errors, 429 and 5xx."""

    def fetch(url: str) -> bytes:
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == retries:
                    raise
            time.sleep(backoff_s * 2**attempt)
        raise AssertionError("unreachable")

    return fetch


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksum(text: str, expected_name: str) -> str:
    """Parse a ``sha256sum``-format line (``<hex>  <name>``; the archive omits the final newline)."""
    parts = text.strip().split()
    if len(parts) != 2 or len(parts[0]) != 64 or parts[1].lstrip("*") != expected_name:
        raise ChecksumError(f"unexpected CHECKSUM content for {expected_name}: {text[:120]!r}")
    return parts[0].lower()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


@dataclass
class ManifestRow:
    symbol: str
    interval: str
    file: str
    url: str
    sha256: str
    size_bytes: int
    last_modified: str
    downloaded_at: str


def fetch_verified(url: str, dest: Path, fetch: Callable[[str], bytes]) -> tuple[str, bool]:
    """Ensure ``dest`` holds the file at ``url`` with a verified checksum.

    Returns ``(sha256, downloaded)``; ``downloaded`` is False when a verified copy already existed.
    The ``.CHECKSUM`` sidecar is stored next to the file.
    """
    ck_path = dest.with_name(dest.name + ".CHECKSUM")
    ck_text = fetch(url + ".CHECKSUM").decode("ascii")
    want = parse_checksum(ck_text, dest.name)
    if dest.exists() and sha256_of(dest) == want:
        if not ck_path.exists():
            _atomic_write(ck_path, ck_text.encode("ascii"))
        return want, False
    data = fetch(url)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ChecksumError(f"{dest.name}: sha256 {got} != published {want}")
    _atomic_write(dest, data)
    _atomic_write(ck_path, ck_text.encode("ascii"))
    return want, True


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    """Merge ``rows`` into the manifest at ``path`` (keyed by file name) and rewrite it atomically.

    A file that was re-verified but not re-downloaded keeps its original ``downloaded_at``.
    """
    existing: dict[str, dict] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            existing = {r["file"]: r for r in csv.DictReader(fh)}
    for r in rows:
        d = {k: str(v) for k, v in asdict(r).items()}
        old = existing.get(r.file)
        if old and old["sha256"] == d["sha256"] and d["downloaded_at"] == "":
            d["downloaded_at"] = old["downloaded_at"]
        existing[r.file] = d
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])
    os.replace(tmp, path)


def manifest_hash(path: Path) -> str:
    """Hash of (file, sha256) pairs only, so re-downloading identical files does not change it."""
    with open(path, newline="", encoding="utf-8") as fh:
        pairs = sorted((r["file"], r["sha256"]) for r in csv.DictReader(fh))
    return hashlib.sha256("\n".join(f"{f} {s}" for f, s in pairs).encode()).hexdigest()


def verify_against_manifest(path: Path, raw_root: Path) -> list[str]:
    """Re-hash every file listed in the manifest. Returns the files that are missing or differ."""
    bad = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = raw_root / r["interval"] / r["symbol"] / r["file"]
            if not f.exists() or sha256_of(f) != r["sha256"]:
                bad.append(r["file"])
    return bad
