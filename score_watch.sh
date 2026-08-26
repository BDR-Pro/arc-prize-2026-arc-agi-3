#!/usr/bin/env bash
# Poll Kaggle for NEW scored submissions; print one SCORED line per new
# completed score. Used by a persistent Monitor so a new score wakes the
# session for the next iteration. Poll every 5 min (scores take hours;
# Kaggle rate-limits favor infrequent polls).
set -u
KG=/home/bader/kaggle/arc/ARC-AGI-3-Kaggle-Starter/.venv/bin/kaggle
TOK=$(cat /home/bader/.kaggle/access_token)
SEEN=/tmp/arc_scored_seen.txt
touch "$SEEN"

while true; do
  curl -s -H "Authorization: Bearer $TOK" \
    "https://www.kaggle.com/api/v1/competitions/submissions/list/arc-prize-2026-arc-agi-3" \
    2>/dev/null | python3 -c '
import sys, json
try:
    subs = json.load(sys.stdin)
except Exception:
    subs = []
seen = set(open("/tmp/arc_scored_seen.txt").read().split())
for s in subs if isinstance(subs, list) else []:
    ref = str(s.get("ref") or s.get("refNullable") or "")
    score = s.get("publicScoreNullable")
    status = str(s.get("statusNullable") or "")
    if ref and score not in (None, "") and ref not in seen:
        desc = (s.get("descriptionNullable") or "")[:70]
        print(f"SCORED {ref} score={score} :: {desc}", flush=True)
        open("/tmp/arc_scored_seen.txt", "a").write(ref + "\n")
' || true
  sleep 300
done
