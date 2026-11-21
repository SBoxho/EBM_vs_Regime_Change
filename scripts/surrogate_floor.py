"""Surrogate-referenced noise floor (D-57). Code: ``ebmregime.surfloor`` (and ``surfloor_power``).

Usage: python scripts/surrogate_floor.py fit    [--workers 16]   # seed-only fits, surrogates 0-19
       python scripts/surrogate_floor.py pairs  [--workers 16]   # control arms: surrogates + real shuf
       python scripts/surrogate_floor.py check                   # D-57 checks (a) and (b)
       python scripts/surrogate_floor.py power  [--workers 16]   # D-57 check (c), synthetic break
       python scripts/surrogate_floor.py real   [--workers 16]   # real dir/vol; only after (a)-(c) pass
       add --gen perm for the leakage-free block-permutation surrogate (D-60)
       python scripts/surrogate_floor.py check-rank              # D-62 gates (a)-(d) on the stored perm pairs
       python scripts/surrogate_floor.py real-rank [--workers 16] # D-62 real dir/vol; only after its gates pass
       python scripts/surrogate_floor.py ext-fit | ext-pairs | ext-summary  # D-65: vol perm surrogates 20-119
       python scripts/surrogate_floor.py block-fit | block-pairs | block-summary  # D-67 (2): block length
       python scripts/surrogate_floor.py snr                      # D-67 (1): H1-dir signal condition
"""

import argparse
import json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "pairs", "check", "check-rank", "power", "real", "real-rank", "ext-fit", "ext-pairs", "ext-summary", "block-fit", "block-pairs", "block-summary", "snr"])
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
    elif a.cmd.startswith("block-"):
        from ebmregime import surfloor_block as B
        {"block-fit": lambda: B.cmd_fit(a.workers), "block-pairs": lambda: B.cmd_pairs(a.workers),
         "block-summary": B.summarise}[a.cmd]()
    elif a.cmd == "snr":
        from ebmregime import surfloor_snr as N
        N.cmd_snr(a.workers)
    elif a.cmd.startswith("ext-"):
        from ebmregime import surfloor_ext as E
        {"ext-fit": lambda: E.cmd_fit(a.workers), "ext-pairs": lambda: E.cmd_pairs(a.workers),
         "ext-summary": E.summarise}[a.cmd]()
    elif a.cmd == "real-rank":
        S.cmd_real_rank(a.workers)
    else:
        if perm:
            raise SystemExit("for the perm generator use real-rank (D-62)")
        S.cmd_real(a.workers)
