import hashlib

import pytest

from ebmregime.data.download import (ChecksumError, ManifestRow, fetch_verified, manifest_hash,
                                     parse_checksum, verify_against_manifest, write_manifest)

DATA = b"zip-bytes"
SHA = hashlib.sha256(DATA).hexdigest()


def fake_fetch(store, calls):
    def fetch(url):
        calls.append(url)
        return store[url]
    return fetch


def test_parse_checksum_without_trailing_newline():
    assert parse_checksum(f"{SHA}  F.zip", "F.zip") == SHA
    with pytest.raises(ChecksumError):
        parse_checksum(f"{SHA}  OTHER.zip", "F.zip")


def test_fetch_verifies_and_skips_when_present(tmp_path):
    store = {"u/F.zip": DATA, "u/F.zip.CHECKSUM": f"{SHA}  F.zip".encode()}
    calls = []
    dest = tmp_path / "F.zip"
    assert fetch_verified("u/F.zip", dest, fake_fetch(store, calls)) == (SHA, True)
    assert dest.read_bytes() == DATA and (tmp_path / "F.zip.CHECKSUM").exists()
    calls.clear()
    assert fetch_verified("u/F.zip", dest, fake_fetch(store, calls)) == (SHA, False)
    assert calls == ["u/F.zip.CHECKSUM"]  # zip not downloaded again


def test_corrupt_download_is_rejected_and_not_written(tmp_path):
    store = {"u/F.zip": b"corrupted", "u/F.zip.CHECKSUM": f"{SHA}  F.zip".encode()}
    with pytest.raises(ChecksumError):
        fetch_verified("u/F.zip", tmp_path / "F.zip", fake_fetch(store, []))
    assert not (tmp_path / "F.zip").exists()


def test_corrupt_local_file_is_replaced(tmp_path):
    dest = tmp_path / "F.zip"
    dest.write_bytes(b"stale")
    store = {"u/F.zip": DATA, "u/F.zip.CHECKSUM": f"{SHA}  F.zip".encode()}
    assert fetch_verified("u/F.zip", dest, fake_fetch(store, []))[1] is True
    assert dest.read_bytes() == DATA


def test_manifest_merge_hash_and_verify(tmp_path):
    raw = tmp_path / "raw"
    (raw / "1h" / "X").mkdir(parents=True)
    (raw / "1h" / "X" / "F.zip").write_bytes(DATA)
    m = tmp_path / "manifest.csv"
    write_manifest(m, [ManifestRow("X", "1h", "F.zip", "u", SHA, 9, "lm", "2026-11-08T00:00:00Z")])
    h1 = manifest_hash(m)
    # Re-verification without a new download keeps the original download time and the hash.
    write_manifest(m, [ManifestRow("X", "1h", "F.zip", "u", SHA, 9, "lm", "")])
    assert "2026-11-08T00:00:00Z" in m.read_text() and manifest_hash(m) == h1
    assert verify_against_manifest(m, raw) == []
    (raw / "1h" / "X" / "F.zip").write_bytes(b"tampered")
    assert verify_against_manifest(m, raw) == ["F.zip"]
