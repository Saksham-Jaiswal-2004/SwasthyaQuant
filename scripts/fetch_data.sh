#!/usr/bin/env bash
# The Kaggle Heart Failure Prediction dataset needs an accepted licence click,
# so it cannot be fetched unattended. Two supported routes:
#
#   A) Kaggle CLI (needs ~/.kaggle/kaggle.json)
#        pip install kaggle
#        kaggle datasets download -d fedesoriano/heart-failure-prediction -p data/raw --unzip
#
#   B) Manual
#        Download from kaggle.com/datasets/fedesoriano/heart-failure-prediction
#        and place heart.csv at data/raw/heart.csv
#
# Then verify -- do not skip this, the audit catches the cholesterol=0 trap:
#        make audit
set -euo pipefail
TARGET="data/raw/heart.csv"
if [[ -f "$TARGET" ]]; then
  echo "found $TARGET  ($(wc -l < "$TARGET") lines including header)"
  echo "next: make audit"
else
  echo "MISSING $TARGET"
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
