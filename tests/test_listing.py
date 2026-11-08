from ebmregime.data.listing import list_prefix, month_of, monthly_zips, parse_list_page

NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def page(keys, truncated=False, extra_checksum_tags=False):
    items = []
    for k in keys:
        ck = ("<ChecksumAlgorithm>CRC64NVME</ChecksumAlgorithm><ChecksumType>FULL_OBJECT</ChecksumType>"
              if extra_checksum_tags else "")
        items.append(f"<Contents><Key>{k}</Key><LastModified>2025-02-03T00:00:00.000Z</LastModified>"
                     f'<ETag>"abc"</ETag>{ck}<Size>123</Size><StorageClass>STANDARD</StorageClass></Contents>')
    return (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult {NS}><Name>b</Name>'
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>{''.join(items)}"
            "</ListBucketResult>").encode()


def test_parser_keeps_objects_with_extra_checksum_elements():
    # The 2025-02+ objects carry extra elements between ETag and Size; a regex parser dropped them.
    objs, truncated, _ = parse_list_page(page(["p/X-1h-2025-02.zip"], extra_checksum_tags=True))
    assert [o.key for o in objs] == ["p/X-1h-2025-02.zip"] and objs[0].size == 123
    assert truncated is False


def test_pagination_uses_last_key_as_marker():
    pages = {None: page(["p/X-1h-2024-01.zip", "p/X-1h-2024-02.zip"], truncated=True),
             "p/X-1h-2024-02.zip": page(["p/X-1h-2024-03.zip"])}
    seen = []

    def fetch(url):
        marker = url.split("marker=")[1].replace("%2F", "/") if "marker=" in url else None
        seen.append(marker)
        return pages[marker]

    objs = list_prefix("https://bucket", "p/", fetch)
    assert [o.filename for o in objs] == ["X-1h-2024-01.zip", "X-1h-2024-02.zip", "X-1h-2024-03.zip"]
    assert seen == [None, "p/X-1h-2024-02.zip"]


def test_monthly_zips_filters_sorts_and_cuts():
    objs, _, _ = parse_list_page(page([
        "p/X-1h-2024-02.zip", "p/X-1h-2024-01.zip", "p/X-1h-2024-01.zip.CHECKSUM",
        "p/part-00000-0fae7358-c000.zip",  # stray Spark output seen in the archive
        "p/X-1h-2024-03.zip"]))
    kept = monthly_zips(objs, last_month="2024-02")
    assert [o.filename for o in kept] == ["X-1h-2024-01.zip", "X-1h-2024-02.zip"]
    assert month_of("BTCUSDT-15m-2025-01.zip") == "2025-01"
