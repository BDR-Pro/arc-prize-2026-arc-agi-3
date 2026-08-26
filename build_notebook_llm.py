"""Build the GPU LLM-agent Kaggle notebook + metadata (separate kernel).

Produces notebooks_llm/submission.ipynb and kernel-metadata.json wired to:
  - GPU enabled
  - qwen-lm/qwen2.5 3b-instruct attached
  - my_agent_llm.py bundle as the agent, hf/transformers backend
The programmatic v79 floor is inside the bundle, so a model-load failure
degrades gracefully to v79 (never worse than 0.27).
"""
import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
AGENT = (HERE / "my_agent_llm.py").read_text(encoding="utf-8")
OUTDIR = HERE / "notebooks_llm"
OUTDIR.mkdir(exist_ok=True)

MODEL_MOUNT = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"


def code(src):
    return {"cell_type": "code", "metadata": {"trusted": True},
            "outputs": [], "execution_count": None, "source": src}


install = code(
    "!pip install --no-index --find-links "
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels "
    "arc-agi python-dotenv\n"
    "import os\n"
    "# locate the attached qwen model dir (mount path can vary by version)\n"
    "import glob\n"
    "cands = glob.glob('/kaggle/input/**/7b-instruct/**/config.json', recursive=True)\n"
    "MODEL_DIR = os.path.dirname(cands[0]) if cands else " + repr(MODEL_MOUNT) + "\n"
    "print('MODEL_DIR =', MODEL_DIR, '| exists:', os.path.isdir(MODEL_DIR))"
)

write_agent = code("%%writefile /tmp/my_agent.py\n" + AGENT)

run = code(dedent("""\
    import os
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
              --retry-max-time 600 http://gateway:8001/api/games
        !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
               /kaggle/working/ARC-AGI-3-Agents
        !cp /tmp/my_agent.py \\
            /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
        with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py','w') as f:
            f.write(\"\"\"from typing import Type
    from dotenv import load_dotenv
    from .agent import Agent, Playback
    from .swarm import Swarm
    from .templates.random_agent import Random
    from .templates.my_agent import MyAgent
    load_dotenv()
    AVAILABLE_AGENTS: dict[str, Type[Agent]] = {'random': Random, 'myagent': MyAgent}
    \"\"\")
        with open('/kaggle/working/ARC-AGI-3-Agents/.env','w') as f:
            f.write(\"\"\"SCHEME=http
    HOST=gateway
    PORT=8001
    ARC_API_KEY=test-key-123
    ARC_BASE_URL=http://gateway:8001/
    OPERATION_MODE=online
    ENVIRONMENTS_DIR=
    RECORDINGS_DIR=/kaggle/working/server_recording
    \"\"\")
        env = (
            'MPLBACKEND=agg '
            'ARC_LLM_BACKEND=hf '
            'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
            f'ARC_LLM_MODEL={MODEL_DIR} '
        )
        !cd /kaggle/working/ARC-AGI-3-Agents && {env} python main.py --agent myagent
    """))

dummy = code(dedent("""\
    import os
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        import pandas as pd
        pd.DataFrame([['1_0','1',True,1]],
            columns=['row_id','game_id','end_of_game','score']
        ).to_parquet('/kaggle/working/submission.parquet', index=False)
    """))

nb = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False,
                   "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
    },
    "nbformat_minor": 4, "nbformat": 4,
    "cells": [
        {"cell_type": "markdown", "metadata": {},
         "source": "# ARC-AGI-3 LLM Agent (Qwen2.5-7B + v79 programmatic floor)\n"
                   "GPU kernel. LLM proposes short action sequences; the proven\n"
                   "programmatic agent is the hard floor on any model failure."},
        install, write_agent, run, dummy,
    ],
}
(OUTDIR / "submission.ipynb").write_text(json.dumps(nb, indent=1))

meta = {
    "id": "baderalotaibi11/arc-agi-3-llm-agent",
    "title": "ARC-AGI-3 LLM Agent",
    "code_file": "submission.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": [], "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["qwen-lm/qwen2.5/transformers/7b-instruct/1"],
}
(OUTDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", OUTDIR / "submission.ipynb", "and kernel-metadata.json")
