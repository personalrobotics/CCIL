#!/usr/bin/env python3
"""Compare a regenerated oracle.json with the committed one (docs/experiments/pendulum-oracle.md, "Regeneration").

    python tools/pendulum_oracle/compare.py docs/fixtures/evidence/pendulum-oracle/oracle.json NEW/oracle.json

Exits 0 only when the two documents are equal everywhere except the declared provenance and timing fields
(VOLATILE): wall time and the code commit and dirty flag of the run that wrote them. The numerical environment
(numpy, scipy, python) is compared like every other field, so a regeneration under other versions is reported as a
difference, not excused. Floats are compared exactly: the oracle is deterministic (version 6.1 reproduced
version 6's L1 and L2 results exactly).
"""
from __future__ import annotations

import json
import sys

VOLATILE = {("seconds",), ("environment", "code_commit"), ("environment", "code_dirty")}


def diff(a, b, path=()) -> list[str]:
    if path in VOLATILE:
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"/{'/'.join(path + (k,))}: only in {'committed' if k in a else 'regenerated'}"
               for k in sorted(set(a) ^ set(b)) if path + (k,) not in VOLATILE]
        return out + [d for k in sorted(set(a) & set(b)) for d in diff(a[k], b[k], path + (k,))]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"/{'/'.join(path)}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, path + (str(i),))]
    return [] if a == b and type(a) is type(b) else [f"/{'/'.join(path)}: {a!r} != {b!r}"]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    committed, regenerated = (json.load(open(p)) for p in sys.argv[1:])
    d = diff(committed, regenerated)
    print("identical (except " + ", ".join("/".join(v) for v in sorted(VOLATILE)) + ")" if not d
          else f"differs in {len(d)} fields:\n" + "\n".join(d[:50]))
    return 0 if not d else 1


if __name__ == "__main__":
    sys.exit(main())
