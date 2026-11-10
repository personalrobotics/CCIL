#!/usr/bin/env python3
"""Check that every relative Markdown link, its #anchor, and every link to this repository's files resolves.

    python3 scripts/check_links.py

Scans the tracked Markdown files outside docs/fixtures/evidence. Exits 1 and lists each broken link.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s]+)\)")
SELF = re.compile(r"https://github\.com/personalrobotics/CCIL/(?:blob|tree)/[^/]+/([^)#\s]+)")


def anchors(path: Path) -> set:
    """GitHub's heading anchors: lower case, punctuation dropped, spaces to hyphens."""
    out = set()
    for line in path.read_text().splitlines():
        m = re.match(r"#+\s+(.*)", line)
        if m:
            out.add(re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-"))
    return out


def main() -> int:
    files = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.md"], capture_output=True, text=True,
                           check=True).stdout.split()
    broken = []
    for rel in files:
        if rel.startswith("docs/fixtures/evidence/") or not (REPO / rel).exists():
            continue
        doc = REPO / rel
        for target in LINK.findall(doc.read_text()):
            own = SELF.match(target)
            if own:
                if not (REPO / own.group(1)).exists():
                    broken.append(f"{rel}: {target}")
                continue
            if re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = target.partition("#")
            dest = (doc.parent / path).resolve() if path else doc
            if not dest.exists():
                broken.append(f"{rel}: {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                broken.append(f"{rel}: {target} (no such heading)")
    for b in broken:
        print(b)
    print(f"{len(files)} Markdown files, {len(broken)} broken links")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
