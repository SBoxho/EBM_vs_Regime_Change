"""List files in the Binance Vision S3 bucket.

The site's own ``list.js`` queries the bucket's S3 ListObjects (V1) endpoint. Two traps:

* Objects uploaded from 2025-02 onward carry extra ``<ChecksumAlgorithm>`` / ``<ChecksumType>``
  elements. A regex parser that expects ``<ETag>`` to be followed by ``<Size>`` silently drops them,
  so we always use a real XML parser.
* Pagination: ``IsTruncated`` is true when more keys exist. Without a delimiter S3 V1 returns no
  ``NextMarker``; the last returned key is then the next ``marker``.
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_MONTHLY = re.compile(r"(?P<sym>[A-Z0-9]+)-(?P<iv>\w+)-(?P<y>\d{4})-(?P<m>\d{2})\.zip$")

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: str  # ISO 8601, as reported by S3

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]


def klines_prefix(market: str, symbol: str, interval: str) -> str:
    """Prefix of the monthly kline files, e.g. ``data/spot/monthly/klines/BTCUSDT/1h/``."""
    return f"data/{market}/monthly/klines/{symbol}/{interval}/"


def parse_list_page(xml_bytes: bytes) -> tuple[list[S3Object], bool, str | None]:
    """Parse one ListObjects page. Returns ``(objects, is_truncated, next_marker)``."""
    root = ET.fromstring(xml_bytes)
    objs = []
    for c in root.findall("s3:Contents", S3_NS):
        objs.append(
            S3Object(
                key=c.findtext("s3:Key", namespaces=S3_NS),
                size=int(c.findtext("s3:Size", namespaces=S3_NS)),
                last_modified=c.findtext("s3:LastModified", namespaces=S3_NS),
            )
        )
    truncated = (root.findtext("s3:IsTruncated", namespaces=S3_NS) or "false").lower() == "true"
    next_marker = root.findtext("s3:NextMarker", namespaces=S3_NS)
    if truncated and not next_marker and objs:
        next_marker = objs[-1].key
    return objs, truncated, next_marker


def list_prefix(list_url: str, prefix: str, fetch: Fetch, max_pages: int = 100) -> list[S3Object]:
    """List every object under ``prefix``, following pagination."""
    out: list[S3Object] = []
    marker: str | None = None
    for _ in range(max_pages):
        q = {"prefix": prefix}
        if marker:
            q["marker"] = marker
        objs, truncated, marker = parse_list_page(fetch(f"{list_url}?{urllib.parse.urlencode(q)}"))
        out.extend(objs)
        if not truncated:
            return out
    raise RuntimeError(f"listing of {prefix!r} exceeded {max_pages} pages")


def monthly_zips(objects: list[S3Object], last_month: str | None = None) -> list[S3Object]:
    """Keep monthly ``.zip`` kline files (not ``.CHECKSUM``, not stray names), sorted by month,
    optionally up to and including ``last_month`` (``"YYYY-MM"``)."""
    keep = []
    for o in objects:
        m = _MONTHLY.search(o.filename)
        if not m:
            continue
        month = f"{m['y']}-{m['m']}"
        if last_month is None or month <= last_month:
            keep.append((month, o))
    return [o for _, o in sorted(keep, key=lambda t: t[0])]


def month_of(filename: str) -> str:
    """``"BTCUSDT-1h-2024-12.zip"`` -> ``"2024-12"``."""
    m = _MONTHLY.search(filename)
    if not m:
        raise ValueError(f"not a monthly kline file name: {filename!r}")
    return f"{m['y']}-{m['m']}"
