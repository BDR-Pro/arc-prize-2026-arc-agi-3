"""ARC-AGI-3 LLM-PRIMARY capability test for Colab (transformers backend).

We use plain transformers (NOT vLLM) here on purpose: vLLM's server kept
dying on Colab's Python 3.13, and for a SEQUENTIAL eval (one game at a time)
we don't need vLLM's concurrency -- that only matters for the 110-concurrent
Kaggle deployment. This path is the one that already ran end-to-end.

The question: can a real ~14B model, DRIVING the game, crack levels the
programmatic floor scores ~0 on -- while preserving the floor's fast wins?
If yes on even a couple of games, reasoning transfers to the private set and
we invest in the vLLM/Kaggle deployment. If no, we tune the model/prompt.

HOW TO USE (Colab): GPU runtime (L4 24GB -> 14B; A100 -> 14B; T4 -> 7B).
Run the cell, paste the RESULT block back to Claude Code.
"""
import glob
import os
import subprocess
import sys
import time


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


# ---- 0. GPU ---------------------------------------------------------------
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

# ---- 1. code + games: clone/refresh the repo (always latest) --------------
REPO = "https://github.com/BDR-Pro/arc-prize-2026-arc-agi-3"
DEST = "/content/arc-prize-2026-arc-agi-3"
if os.path.isdir("arc_games"):
    print("running from an existing checkout:", os.getcwd())
elif os.path.isdir(DEST + "/arc_games"):
    os.chdir(DEST)
    sh("git pull --ff-only")
else:
    sh(f"git clone --depth 1 {REPO} {DEST}")
    os.chdir(DEST)
print("code+games at:", os.getcwd(), "| games:", len(glob.glob("arc_games/*/")))

# ---- 2. deps: arc engine + transformers stack ----------------------------
sh("pip -q install arc-agi transformers accelerate 2>&1 | tail -2")
sh("pip -q install autoawq 2>&1 | tail -2")   # for 4-bit AWQ models
try:
    import awq  # noqa: F401
    AWQ_OK = True
except Exception as e:  # noqa: BLE001
    AWQ_OK = False
    print("autoawq unavailable ->", repr(e), "(will use an unquantized model)")

# ---- 3. pick model by VRAM (+ AWQ availability) --------------------------
import torch
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if os.environ.get("ARC_LLM_MODEL"):
    MODEL = os.environ["ARC_LLM_MODEL"]
elif AWQ_OK and gb >= 20:
    MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"       # ~9GB, best capability here
elif gb >= 20:
    MODEL = "Qwen/Qwen2.5-7B-Instruct"            # fp16 ~14GB, fits L4/A100
else:
    MODEL = "Qwen/Qwen2.5-3B-Instruct"            # T4 fallback
print(f"GPU {gb:.0f}GB | autoawq={AWQ_OK} -> model {MODEL}")

# ---- 4. backend=hf; PRELOAD the model once and warm-test it --------------
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "llm"))
os.environ["ARC_LLM_BACKEND"] = "hf"


def preload(model_id):
    """Load model_id into the shared cache; return a warm-up reply."""
    os.environ["ARC_LLM_MODEL"] = model_id
    cache = getattr(sys, "_ARC_LLM_SHARED", None)
    if cache:
        cache.pop("hf", None)            # evict any prior/failed load
    import importlib
    import llm_client
    importlib.reload(llm_client)
    c = llm_client.make_client()
    return c.chat("You are a test.", "Reply with exactly: OK", max_tokens=8)


try:
    print("warm-up reply:", repr(preload(MODEL)))
except Exception as e:  # noqa: BLE001
    print("!!! model load FAILED:", repr(e), "-> falling back to 7B fp16")
    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    print("warm-up reply:", repr(preload(MODEL)))
print("model ready:", MODEL)

# ---- 5. eval floor vs LLM-primary on a focused subset --------------------
# default: a few games the floor scores ~0 on (room for the LLM) + two of
# the floor's fast wins (ar25, vc33) to confirm the LLM doesn't break them.
DEFAULT_SET = "dc22,ka59,sb26,su15,cd82,ar25,vc33"
sel = os.environ.get("ARC_EVAL_GAMES", DEFAULT_SET)
from pathlib import Path
ALL = sorted(d.name for d in Path("arc_games").iterdir() if d.is_dir())
GAMES = ALL if sel == "all" else [g for g in sel.split(",") if g in ALL]

MAXACT = int(os.environ.get("ARC_EVAL_MAXACT", "2000"))
os.environ.setdefault("ARC_LLM_MAX_CALLS", "40")   # per-game LLM budget
os.environ.setdefault("ARC_LLM_MAX_TOKENS", "384")
os.environ.setdefault("ARC_LLM_FLOOR_OPENING", "100")
os.environ.setdefault("ARC_LLM_DEBUG", "1")        # show the model's replies
print(f"games={GAMES} max_actions={MAXACT} "
      f"llm_calls<={os.environ['ARC_LLM_MAX_CALLS']}")


def eval_agent(agent_path, tag):
    import importlib
    import eval_harness
    importlib.reload(eval_harness)
    tot_score = tot_lv = 0
    per = {}
    for i, g in enumerate(GAMES):
        r = eval_harness.run_game(g, Path(agent_path), MAXACT)
        lv, sc = r.get("levels_completed", 0), r.get("score", 0.0)
        per[g] = (lv, sc)
        tot_lv += lv
        tot_score += sc
        print(f"  [{tag}] {i+1}/{len(GAMES)} {g}: lv={lv} score={sc:.3f}",
              flush=True)
    mean = tot_score / max(len(GAMES), 1)
    print(f"[{tag}] levels={tot_lv} mean={mean:.4f}")
    return tot_lv, mean, per


print("\n=== evaluating programmatic floor (v79) ===")
p_lv, p_mean, p_per = eval_agent("my_agent.py", "PROG")
print(f"\n=== evaluating LLM-PRIMARY ({MODEL}) ===")
l_lv, l_mean, l_per = eval_agent("my_agent_primary.py", "LLM")

print("\n" + "=" * 52)
print("RESULT (paste this back to Claude Code):")
print(f"  model={MODEL}  gpu={gb:.0f}GB  games={len(GAMES)}")
print(f"  PROG floor  : levels={p_lv} mean={p_mean:.4f}")
print(f"  LLM primary : levels={l_lv} mean={l_mean:.4f}")
print(f"  LLM delta   : {l_mean - p_mean:+.4f} mean, {l_lv - p_lv:+d} levels")
print("  per-game (game: prog_lv/score -> llm_lv/score):")
for g in GAMES:
    mark = ""
    if l_per[g][1] > p_per[g][1] + 1e-9:
        mark = "  <== LLM WIN"
    elif l_per[g][1] < p_per[g][1] - 1e-9:
        mark = "  <== regressed"
    print(f"    {g}: {p_per[g][0]}/{p_per[g][1]:.3f} -> "
          f"{l_per[g][0]}/{l_per[g][1]:.3f}{mark}")
print("=" * 52)
