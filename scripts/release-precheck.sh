#!/usr/bin/env bash
# Build and exercise the exact distributions that a release would publish.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain)" ]]; then
    echo "release precheck requires a clean worktree" >&2
    exit 1
fi

cleanup=false
if [[ $# -eq 0 ]]; then
    out=$(mktemp -d "${TMPDIR:-/tmp}/ccil-release.XXXXXX")
    cleanup=true
elif [[ $# -eq 1 ]]; then
    out=$1
    if [[ -e "$out" ]]; then
        echo "output already exists: $out" >&2
        exit 1
    fi
else
    echo "usage: $0 [NEW_OUTPUT_DIRECTORY]" >&2
    exit 2
fi

if $cleanup; then
    trap 'rm -rf "$out"' EXIT
fi

uv lock --check
uv run python tools/m2_packaging/check.py --out "$out/package"
uv run python tools/golden_path/check.py --out "$out/golden"
uvx --from twine==6.2.0 twine check "$out"/package/dist1/*.whl "$out"/package/dist1/*.tar.gz

echo "release precheck passed for prlccil $(uv version --short)"
echo "verified distributions: $out/package/dist1"
