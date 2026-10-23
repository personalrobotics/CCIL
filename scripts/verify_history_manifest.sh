#!/usr/bin/env bash
# Compare every ref recorded in docs/history/refs.tsv with the live remote.
# Also reports upstream heads and tags that the manifest does not record.
# A branch that only moved forward (recorded commit still in its history) is
# ADVANCED and passes; tags must match exactly.
# Exits non-zero if any recorded ref was rewritten, moved, or disappeared.
set -euo pipefail

manifest="${1:-$(dirname "$0")/../docs/history/refs.tsv}"
upstream="https://github.com/personalrobotics/CCIL"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# One ls-remote per repository, cached by a hash of its URL.
refs_of() {
  local cache="$tmp/$(printf '%s' "$1" | cksum | cut -d' ' -f1)"
  [[ -f "$cache" ]] || git ls-remote "$1" >"$cache"
  cat "$cache"
}

# True if the branch's live tip still contains the recorded commit.
advanced() {
  local repo="$tmp/objects"
  [[ -d "$repo" ]] || git init -q --bare "$repo"
  git -C "$repo" fetch -q "$1" "+$2:refs/check/$(printf '%s' "$1$2" | cksum | cut -d' ' -f1)" &&
    git -C "$repo" merge-base --is-ancestor "$3" FETCH_HEAD 2>/dev/null
}

status=0
while IFS=$'\t' read -r url ref sha; do
  [[ -z "$url" || "$url" == \#* ]] && continue
  actual="$(refs_of "$url" | awk -v r="$ref" '$2 == r { print $1 }')"
  if [[ -z "$actual" ]]; then
    echo "MISSING    $url $ref (expected $sha)"; status=1
  elif [[ "$actual" != "$sha" && "$ref" == refs/heads/* ]] && advanced "$url" "$ref" "$sha"; then
    echo "ADVANCED   $url $ref $sha -> $actual"
  elif [[ "$actual" != "$sha" ]]; then
    echo "MOVED      $url $ref $sha -> $actual"; status=1
  else
    echo "OK         $url $ref $sha"
  fi
done <"$manifest"

refs_of "$upstream" | while read -r _ ref; do
  [[ "$ref" == refs/heads/* || "$ref" == refs/tags/* ]] || continue
  awk -F'\t' -v u="$upstream" -v r="$ref" '$1 == u && $2 == r { found = 1 } END { exit !found }' "$manifest" \
    || echo "UNRECORDED $upstream $ref"
done

exit "$status"
