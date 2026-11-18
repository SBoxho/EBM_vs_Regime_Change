"""Surrogate-referenced noise floor (D-57). Code: ``ebmregime.surfloor`` (and ``surfloor_power``).

Usage: python scripts/surrogate_floor.py fit    [--workers 16]   # seed-only fits, surrogates 0-19
       python scripts/surrogate_floor.py pairs  [--workers 16]   # control arms: surrogates + real shuf
       python scripts/surrogate_floor.py check                   # D-57 checks (a) and (b)
       python scripts/surrogate_floor.py power  [--workers 16]   # D-57 check (c), synthetic break
       python scripts/surrogate_floor.py real   [--workers 16]   # real dir/vol; only after (a)-(c) pass
"""

import argparse
import json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "pairs", "check", "power", "real"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from ebmregime import surfloor as S
    if a.cmd == "fit":
        S.cmd_fit(a.workers)
    elif a.cmd == "pairs":
        S.cmd_pairs(S.control_arms(), a.workers, "control_pairs")
    elif a.cmd == "check":
        S.cmd_check()
    elif a.cmd == "power":
        from ebmregime.surfloor_power import cmd_power
        cmd_power(a.workers)
    else:
        S.cmd_real(a.workers)
