#!/usr/bin/env bash
set -euo pipefail

# Reproducible build for the USTB 2026 template preview.
# Run from this file's directory or from the repository root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREVIEW="$ROOT/thesis/latex/ustb_2026_preview"

# Prefer an explicitly supplied TeX installation.  The project-local TinyTeX
# tree is supported for the original server environment but is intentionally
# not part of the GitHub archive.
TEXROOT="${TEXROOT:-$ROOT/.TinyTeX}"
if [[ -x "$TEXROOT/bin/x86_64-linux/xelatex" ]]; then
  export PATH="$TEXROOT/bin/x86_64-linux:$PATH"
  export TEXMFHOME="${TEXMFHOME:-$TEXROOT/texmf-local}"
  export TEXMFVAR="${TEXMFVAR:-$TEXROOT/texmf-var}"
fi

command -v xelatex >/dev/null 2>&1 || {
  echo "ERROR: xelatex not found. Install XeLaTeX or set TEXROOT to a TeX Live/TinyTeX installation." >&2
  exit 1
}

cd "$PREVIEW"
perl normalize_content.pl

for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -no-shell-escape -jobname=ustb_mattergen_preview main_mattergen.tex \
    > "build_pass${pass}.log" 2>&1
done

LOG="$PREVIEW/build_pass3.log"
test -s ustb_mattergen_preview.pdf
if grep -q 'Missing character' "$LOG"; then
  echo "ERROR: missing glyphs remain" >&2
  exit 1
fi
if grep -q 'Citation.*undefined\|There were undefined citations' "$LOG"; then
  echo "ERROR: undefined citations remain" >&2
  exit 1
fi
test "$(grep -cF '\bibitem{' contents/bib.tex)" -eq 22
test "$(grep -cF '\contentsline {table}' ustb_mattergen_preview.lot)" -eq 24
test "$(grep -cF '\contentsline {figure}' ustb_mattergen_preview.lof)" -eq 24

echo "Built: $PREVIEW/ustb_mattergen_preview.pdf"
echo "Pages: $(grep -o 'Output written on .* ([0-9][0-9]* pages)' "$LOG" | tail -1)"
