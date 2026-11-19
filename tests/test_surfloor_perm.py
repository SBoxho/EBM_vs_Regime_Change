"""Block-permutation surrogate (D-60): a permutation of source rows, purged at block ends."""

import numpy as np
import pandas as pd

from ebmregime.surfloor_perm import make_perm_surrogate, perm_rows


def test_perm_rows_is_permutation_of_contiguous_blocks():
    src, end_dist = perm_rows(10_000, 252, 3)
    np.testing.assert_array_equal(np.sort(src), np.arange(10_000))
    jumps = np.flatnonzero(np.diff(src) != 1) + 1                 # block starts in surrogate order
    assert 10 < len(jumps) + 1 < 100                              # about 10000/252 blocks
    last_in_block = np.r_[jumps - 1, len(src) - 1]
    ends_at_n = src[last_in_block] == 9_999
    assert (end_dist[last_in_block][~ends_at_n] == 0).all()
    assert (end_dist[src == 9_999] >= 10_000).all()               # the block ending the range is never purged


def test_perm_rows_deterministic_and_seed_dependent():
    a, _ = perm_rows(5000, 252, 1)
    b, _ = perm_rows(5000, 252, 1)
    c, _ = perm_rows(5000, 252, 2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_make_perm_surrogate_purges_label_tails_only():
    ts = pd.date_range("2020-01-01", periods=3000, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(3000.0), "y_dir_h8": np.ones(3000), "y_vol_h24": np.ones(3000)}, index=ts)
    out = make_perm_surrogate(df, ts[0], ts[-1] + pd.Timedelta(hours=1), 252, 5)
    assert out.index.equals(df.index)
    np.testing.assert_array_equal(np.sort(out.x.to_numpy()), np.arange(3000.0))   # every row once
    src = out.x.to_numpy().astype(int)
    _, end_dist = perm_rows(3000, 252, 5)
    np.testing.assert_array_equal(out.y_dir_h8.isna().to_numpy(), end_dist < 8)
    np.testing.assert_array_equal(out.y_vol_h24.isna().to_numpy(), end_dist < 24)
    # no two usable h8 rows from different blocks share label hours
    ok = np.flatnonzero(out.y_dir_h8.notna().to_numpy())
    blocks = np.cumsum(np.r_[True, np.diff(src) != 1])
    s_ok, b_ok = src[ok], blocks[ok]
    order = np.argsort(s_ok)
    s_sorted, b_sorted = s_ok[order], b_ok[order]
    gap = np.diff(s_sorted)
    cross = b_sorted[1:] != b_sorted[:-1]
    assert (gap[cross] >= 8).all()
