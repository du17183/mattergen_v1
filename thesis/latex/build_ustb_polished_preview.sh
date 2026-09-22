#!/usr/bin/env bash
set -euo pipefail

# Reproducible polished-artwork preview.  This script is intentionally
# separate from build_ustb_preview.sh; it never overwrites the original
# artwork or the original 127-page preview.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREVIEW="$ROOT/thesis/latex/ustb_2026_polished_preview"

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
perl polish_latex_fragments.pl

for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -no-shell-escape -jobname=ustb_mattergen_polished main_mattergen.tex \
    > "build_pass${pass}.log" 2>&1
done

LOG="$PREVIEW/build_pass3.log"
test -s ustb_mattergen_polished.pdf
if grep -q 'Missing character' "$LOG"; then
  echo "ERROR: missing glyphs remain" >&2
  exit 1
fi
if grep -q 'Citation.*undefined\|There were undefined citations' "$LOG"; then
  echo "ERROR: undefined citations remain" >&2
  exit 1
fi
test "$(grep -cF '\bibitem{' contents/bib.tex)" -eq 22
test "$(grep -cF '\contentsline {table}' ustb_mattergen_polished.lot)" -eq 24
test "$(grep -cF '\contentsline {figure}' ustb_mattergen_polished.lof)" -eq 24
test "$(rg -c '\\label\{fig:' contents/chap*.tex | awk -F: '{s+=$2} END {print s+0}')" -eq 24
test "$(rg -o --no-filename 'innovation[12]/figures_polished/[^} ]+\.pdf' contents/chap*.tex | sort -u | wc -l)" -eq 12
test "$(rg -o --no-filename 'concept_figures_polished/[^} ]+\.pdf' contents/chap*.tex | sort -u | wc -l)" -eq 6
test "$(rg -c 'release ID:' contents/chap*.tex | awk -F: '{s+=$2} END {print s+0}')" -eq 0

echo "Built: $PREVIEW/ustb_mattergen_polished.pdf"
echo "Pages: $(grep -o 'Output written on .* ([0-9][0-9]* pages)' "$LOG" | tail -1)"
