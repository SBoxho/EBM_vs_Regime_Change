"""Floor fix experiment (D-55): fit candidate floors on the CONTROL arms only, then evaluate G0.

Arms (BTCUSDT, h = 8): real ``shuf``, ``surrogate:0`` dir, ``surrogate:0`` vol. Real dir and vol
targets are never touched. Candidates per window: B, 10 full-n stationary-bootstrap fits; R, 4
partitions into 2 disjoint halves and 2 partitions into 4 disjoint quarters (16 fits). Seed-only and
half-sample (protocol iii) fits come from the slice cache. Code: ``ebmregime.floorexp`` (fits) and
``ebmregime.floorexp_eval`` (evaluation).

Usage: python scripts/floor_experiment.py fit [--workers 16]
       python scripts/floor_experiment.py eval [--workers 16]
       python scripts/floor_experiment.py tail [--workers 16]   # exploratory B-tail (D-57)
"""

import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval", "tail"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "fit":
        from ebmregime.floorexp import cmd_fit
        cmd_fit(a.workers)
    elif a.cmd == "eval":
        from ebmregime.floorexp_eval import cmd_eval
        cmd_eval(a.workers)
    else:
        from ebmregime.floorexp_tail import cmd_tail
        cmd_tail(a.workers)
