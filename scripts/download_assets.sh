#!/usr/bin/env bash
# Download large runtime assets that are .gitignored.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/data"

# fasttext language identification model (124MB)
LID="$REPO_ROOT/data/lid.176.bin"
if [ ! -f "$LID" ]; then
  echo "→ downloading fasttext lid.176.bin"
  curl -L -o "$LID" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
else
  echo "✓ lid.176.bin already present"
fi

echo "✓ assets ready"
