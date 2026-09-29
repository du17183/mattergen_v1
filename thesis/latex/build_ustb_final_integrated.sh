#!/usr/bin/env bash
set -euo pipefail

# Build the final integrated USTB thesis from committed sources. This does not
# run generation/evaluation or rewrite the manuscript and artwork.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREVIEW="$ROOT/thesis/latex/ustb_2026_polished_preview"
TEXROOT="${TEXROOT:-$ROOT/.TinyTeX}"
if [[ -x "$TEXROOT/bin/x86_64-linux/xelatex" ]]; then
  export PATH="$TEXROOT/bin/x86_64-linux:$PATH"
  export TEXMFHOME="${TEXMFHOME:-$TEXROOT/texmf-local}"
  export TEXMFVAR="${TEXMFVAR:-$TEXROOT/texmf-var}"
fi

command -v xelatex >/dev/null 2>&1 || {
  echo "ERROR: XeLaTeX not found; install TeX Live/TinyTeX 2026 or set TEXROOT." >&2
  exit 1
}
FONT="$ROOT/.TinyTeX/texmf-dist/fonts/opentype/public/fandol/FandolSong-Regular.otf"
[[ -f "$FONT" ]] || {
  echo "ERROR: required Fandol font missing at $FONT; the current USTB class uses this relative path." >&2
  exit 1
}

cd "$PREVIEW"
for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -no-shell-escape -jobname=ustb_mattergen_final_integrated main_mattergen.tex \
    > "build_final_integrated_pass${pass}.log" 2>&1
done

PDF=ustb_mattergen_final_integrated.pdf
LOG=build_final_integrated_pass3.log
test -s "$PDF"
if rg -n 'Missing character|There were undefined references|There were undefined citations|Citation.*undefined|Reference.*undefined|File .* not found' "$LOG"; then
  echo "ERROR: unresolved font, citation, reference or file in final build." >&2
  exit 1
fi
test "$(grep -cF '\contentsline {figure}' ustb_mattergen_final_integrated.lof)" -eq 30
test "$(grep -cF '\contentsline {table}' ustb_mattergen_final_integrated.lot)" -eq 29
if command -v pdfinfo >/dev/null 2>&1; then
  test "$(pdfinfo "$PDF" | awk '/^Pages:/ {print $2}')" -eq 137
fi
echo "Built: $PREVIEW/$PDF"
sha256sum "$PDF"
