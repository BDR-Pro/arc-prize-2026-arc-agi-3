"""ARC-AGI-3 LLM-agent GPU eval for Google Colab.

HOW TO USE (Colab, GPU runtime):
  1. Runtime -> Change runtime type -> GPU (T4/L4/A100).
  2. Upload arc_kit.zip (Files pane, or run the upload cell).
  3. Paste this whole file into ONE Colab cell and run it.
It installs vLLM + arc-agi, serves Qwen2.5-7B, and evaluates the real
LLM rescue-agent vs the pure programmatic floor on the 25 public games,
printing both mean scores so we can SEE whether the LLM adds value.

Copy the printed RESULT block back to Claude Code.
"""
import os, sys, subprocess, time, zipfile, glob, json, threading

def sh(cmd):
    print("+", cmd); return subprocess.run(cmd, shell=True)

# ---- 0. GPU check --------------------------------------------------------
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

# ---- 1. deps -------------------------------------------------------------
sh("pip -q install vllm==0.6.3.post1 arc-agi 2>/dev/null || pip -q install vllm arc-agi")

# ---- 2. get the code + games: clone the repo (always latest) -------------
REPO = "https://github.com/BDR-Pro/arc-prize-2026-arc-agi-3"
if os.path.isdir("arc_games"):
    print("running from an existing checkout:", os.getcwd())
elif os.path.isdir("/content/arc-prize-2026-arc-agi-3/arc_games"):
    os.chdir("/content/arc-prize-2026-arc-agi-3")
else:
    sh(f"git clone --depth 1 {REPO} /content/arc-prize-2026-arc-agi-3")
    os.chdir("/content/arc-prize-2026-arc-agi-3")
print("code+games at:", os.getcwd(), "| games:", len(glob.glob("arc_games/*/")))

# ---- 3. pick model by GPU memory ----------------------------------------
import torch
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
MODEL = "Qwen/Qwen2.5-7B-Instruct" if gb >= 20 else "Qwen/Qwen2.5-3B-Instruct"
print(f"GPU {gb:.0f}GB -> model {MODEL}")

# ---- 4. start a vLLM OpenAI server in the background ---------------------
srv = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
     "--model", MODEL, "--port", "8000",
     "--max-model-len", "4096", "--gpu-memory-utilization", "0.85"],
    stdout=open("/content/vllm.log", "w"), stderr=subprocess.STDOUT)
print("waiting for vLLM server...")
import urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
        print("vLLM up"); break
    except Exception:
        time.sleep(5)
else:
    print("vLLM did NOT start; tail of log:")
    sh("tail -40 /content/vllm.log"); raise SystemExit

# ---- 5. eval both agents on the 25 games --------------------------------
os.environ["ARC_LLM_BACKEND"] = "openai"
os.environ["ARC_LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
os.environ["ARC_LLM_MODEL"] = MODEL
sys.path.insert(0, "/content/arc")

def eval_agent(agent_path, tag, max_actions=4000):
    import importlib, eval_harness
    importlib.reload(eval_harness)
    from pathlib import Path
    games = sorted(d.name for d in Path("arc_games").iterdir() if d.is_dir())
    tot_score = tot_lv = 0
    per = {}
    for g in games:
        r = eval_harness.run_game(g, Path(agent_path), max_actions)
        per[g] = (r.get("levels_completed", 0), r.get("score", 0.0))
        tot_lv += r.get("levels_completed", 0)
        tot_score += r.get("score", 0.0)
    mean = tot_score / max(len(games), 1)
    print(f"[{tag}] levels={tot_lv} mean={mean:.4f}")
    return tot_lv, mean, per

print("\n=== evaluating programmatic floor (v79) ===")
p_lv, p_mean, p_per = eval_agent("my_agent.py", "PROG")
print("\n=== evaluating LLM rescue-agent (real 7B via vLLM) ===")
l_lv, l_mean, l_per = eval_agent("my_agent_llm.py", "LLM")

print("\n" + "=" * 50)
print("RESULT (paste this back to Claude Code):")
print(f"  model={MODEL}")
print(f"  PROG floor : levels={p_lv} mean={p_mean:.4f}")
print(f"  LLM agent  : levels={l_lv} mean={l_mean:.4f}")
print(f"  LLM delta  : {l_mean - p_mean:+.4f} mean, {l_lv - p_lv:+d} levels")
print("  per-game (game: prog_lv/prog_score -> llm_lv/llm_score) where changed:")
for g in sorted(p_per):
    if p_per[g] != l_per[g]:
        print(f"    {g}: {p_per[g][0]}/{p_per[g][1]:.3f} -> {l_per[g][0]}/{l_per[g][1]:.3f}")
print("=" * 50)
