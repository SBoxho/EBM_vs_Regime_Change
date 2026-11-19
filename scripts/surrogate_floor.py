"""Surrogate-referenced noise floor (D-57). Code: ``ebmregime.surfloor`` (and ``surfloor_power``).

Usage: python scripts/surrogate_floor.py fit    [--workers 16]   # seed-only fits, surrogates 0-19
       python scripts/surrogate_floor.py pairs  [--workers 16]   # control arms: surrogates + real shuf
       python scripts/surrogate_floor.py check                   # D-57 checks (a) and (b)
       python scripts/surrogate_floor.py power  [--workers 16]   # D-57 check (c), synthetic break
       python scripts/surrogate_floor.py real   [--workers 16]   # real dir/vol; only after (a)-(c) pass
       add --gen perm for the leakage-free block-permutation surrogate (D-60)
       python scripts/surrogate_floor.py check-rank              # D-62 gates (a)-(d) on the stored perm pairs
       python scripts/surrogate_floor.py real-rank [--workers 16] # D-62 real dir/vol; only after its gates pass
"""

import argparse
import json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "pairs", "check", "check-rank", "power", "real", "real-rank"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gen", choices=["replace", "perm"], default="replace")
    a = ap.parse_args()
    from ebmregime import surfloor as S
    from ebmregime import surfloor_perm as Q
    perm = a.gen == "perm"
    if a.cmd == "fit":
        Q.cmd_fit(a.workers) if perm else S.cmd_fit(a.workers)
    elif a.cmd == "pairs":
        Q.cmd_pairs(a.workers) if perm else S.cmd_pairs(S.control_arms(), a.workers, "control_pairs")
    elif a.cmd == "check":
        S.cmd_check("perm_control_pairs", prefix="perm_") if perm else S.cmd_check()
    elif a.cmd == "check-rank":
        out = S.cmd_check_rank()
        c = S.summarise_power_rank()
        print("D-62 gates:", json.dumps({"a": out["a_pass"], "b": out["b_pass"], "c": c["pass"]}))
    elif a.cmd == "power":
        from ebmregime.surfloor_power import cmd_power
        cmd_power(a.workers, gen=a.gen)
    elif a.cmd == "real-rank":
        S.cmd_real_rank(a.workers)
    else:
        if perm:
            raise SystemExit("for the perm generator use real-rank (D-62)")
        S.cmd_real(a.workers)
