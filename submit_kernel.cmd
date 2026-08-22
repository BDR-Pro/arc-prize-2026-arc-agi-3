@echo off
wsl -e bash -lc "cd /home/bader/kaggle/arc && /home/bader/kaggle/arc/ARC-AGI-3-Kaggle-Starter/.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-3 -k baderalotaibi11/arc-agi-3-world-model-agent -v 10 -f submission.parquet -m 'v58 desperation-mode arsenal + 8k time-capped budget' >> /home/bader/submit_log.txt 2>&1"
