@echo off
wsl -e bash -lc "cd /home/bader/kaggle/arc && /home/bader/kaggle/arc/ARC-AGI-3-Kaggle-Starter/.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-3 -k baderalotaibi11/arc-agi-3-llm-agent -v 4 -f submission.parquet -m 'LLM 7B rescue-agent v4 (concurrency-fixed) + v79 floor' >> /home/bader/submit_log.txt 2>&1"
