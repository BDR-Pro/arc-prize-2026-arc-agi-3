"""ARC-AGI-3 LLM-PRIMARY GPU eval for Google Colab (vLLM + AWQ).

Tests the real question: does an LLM DRIVING the game (Duck-style) beat the
programmatic floor on the 25 public games? Because the LLM solves by
reasoning about the board (not memorised public-game tricks), success here
is expected to transfer to the 110 private games -- unlike the heuristic
speed tricks that plateaued at LB 0.27.

HOW TO USE (Colab):
  1. Runtime -> Change runtime type -> GPU. Prefer **L4** (14B) or **A100**
     (32B). T4 falls back to a 7B and is only a smoke test.
  2. Run the cell. It clones the latest code, installs vLLM, serves a
     quantized Qwen2.5 (AWQ) sized to the GPU, and evaluates the LLM-primary
     agent vs the programmatic floor.
  3. Copy the printed RESULT block back to Claude Code.

Everything is auto-generated from this file by build_colab_notebook.py.
"""
import glob
import os
import subprocess
import sys
import time
import urllib.request


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

# ---- 2. deps: arc engine + vLLM ------------------------------------------
sh("pip -q install arc-agi 2>&1 | tail -2")
sh("pip -q install vllm 2>&1 | tail -3")
try:
    import vllm  # noqa: F401
    print("vLLM version:", vllm.__version__)
except Exception as e:  # noqa: BLE001
    print("\n!!! vLLM IMPORT FAILED:", repr(e))
    print("If this is a Python-version wheel issue, tell Claude the Python "
          "version:", sys.version)
    raise SystemExit("vLLM unavailable -- cannot run the LLM-primary eval")

# ---- 3. pick a quantized model by VRAM -----------------------------------
import torch
n_gpu = torch.cuda.device_count()
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if os.environ.get("ARC_LLM_MODEL"):
    MODEL = os.environ["ARC_LLM_MODEL"]
elif gb >= 38:
    MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
elif gb >= 21:
    MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
elif gb >= 15:
    MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
else:
    MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
print(f"GPU {gb:.0f}GB x{n_gpu} -> model {MODEL}")

# ---- 4. launch the vLLM OpenAI server, log to file, poll /health ---------
PORT = 8000
LOG = "/content/vllm_server.log"
cmd = [
    "vllm", "serve", MODEL,            # canonical CLI (stable across versions)
    "--tensor-parallel-size", str(n_gpu),
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.90",
    "--enforce-eager",                 # robust + memory-lean on T4/L4
    "--port", str(PORT),
]
# child MUST be unbuffered or its crash traceback never reaches the log file
child_env = dict(os.environ, PYTHONUNBUFFERED="1",
                 VLLM_LOGGING_LEVEL="DEBUG")


def dump_log(tag):
    print(f"\n!!! {tag}\n----- vllm_server.log -----")
    try:
        with open(LOG, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        print(txt[-6000:] if txt.strip() else "(log is EMPTY)")
    except Exception as e:  # noqa: BLE001
        print("could not read log:", e)
    print("---------------------------")


print("+ launching vLLM server (log ->", LOG + "):")
print("   " + " ".join(cmd))
logf = open(LOG, "w")
try:
    server = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              env=child_env)
except FileNotFoundError:
    # `vllm` script not on PATH -> fall back to the module entrypoint
    print("`vllm` not on PATH; falling back to python -m ...")
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", MODEL, "--tensor-parallel-size", str(n_gpu),
           "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
           "--enforce-eager", "--port", str(PORT)]
    server = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              env=child_env)

def last_log_line():
    try:
        with open(LOG, encoding="utf-8", errors="replace") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        return lines[-1][:120] if lines else "(no output yet)"
    except Exception:                  # noqa: BLE001
        return "(log unreadable)"


BASE = f"http://127.0.0.1:{PORT}/v1"
t_start = time.time()
DEADLINE = t_start + 1800              # up to 30 min for download + load
ready = False
next_beat = t_start + 30
while time.time() < DEADLINE:
    if server.poll() is not None:
        dump_log(f"vLLM server EXITED early (return code {server.returncode})")
        raise SystemExit("vLLM server died during startup")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",
                                    timeout=5) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:                  # noqa: BLE001
        pass
    if time.time() >= next_beat:
        print(f"   ...loading {int(time.time()-t_start)}s | {last_log_line()}",
              flush=True)
        next_beat += 30
    time.sleep(5)
if not ready:
    server.terminate()
    dump_log("vLLM server did not become healthy in time")
    raise SystemExit("vLLM startup timeout")
print("vLLM server is HEALTHY.")

# ---- 5. point the agent at the server ------------------------------------
os.environ["ARC_LLM_BACKEND"] = "openai"
os.environ["ARC_LLM_BASE_URL"] = BASE
os.environ["ARC_LLM_MODEL"] = MODEL
os.environ["ARC_LLM_KEY"] = "EMPTY"
os.environ["ARC_LLM_TIMEOUT"] = "60"

# ---- 6. eval floor vs LLM-primary ----------------------------------------
sys.path.insert(0, os.getcwd())
MAXACT = int(os.environ.get("ARC_EVAL_MAXACT", "4000"))
# deployment-realistic: floor opens each level, LLM takes the hard ones
os.environ.setdefault("ARC_LLM_FLOOR_OPENING", "100")


def eval_agent(agent_path, tag):
    import importlib
    import eval_harness
    importlib.reload(eval_harness)
    from pathlib import Path
    games = sorted(d.name for d in Path("arc_games").iterdir() if d.is_dir())
    tot_score = tot_lv = 0
    per = {}
    for i, g in enumerate(games):
        r = eval_harness.run_game(g, Path(agent_path), MAXACT)
        lv, sc = r.get("levels_completed", 0), r.get("score", 0.0)
        per[g] = (lv, sc)
        tot_lv += lv
        tot_score += sc
        print(f"  [{tag}] {i+1}/{len(games)} {g}: lv={lv} score={sc:.3f}",
              flush=True)
    mean = tot_score / max(len(games), 1)
    print(f"[{tag}] levels={tot_lv} mean={mean:.4f}")
    return tot_lv, mean, per


try:
    print("\n=== evaluating programmatic floor (v79) ===")
    p_lv, p_mean, p_per = eval_agent("my_agent.py", "PROG")
    print(f"\n=== evaluating LLM-PRIMARY (opening="
          f"{os.environ['ARC_LLM_FLOOR_OPENING']}, model {MODEL}) ===")
    l_lv, l_mean, l_per = eval_agent("my_agent_primary.py", "LLM")

    print("\n" + "=" * 52)
    print("RESULT (paste this back to Claude Code):")
    print(f"  model={MODEL}  gpu={gb:.0f}GBx{n_gpu}")
    print(f"  PROG floor  : levels={p_lv} mean={p_mean:.4f}")
    print(f"  LLM primary : levels={l_lv} mean={l_mean:.4f}")
    print(f"  LLM delta   : {l_mean - p_mean:+.4f} mean, "
          f"{l_lv - p_lv:+d} levels")
    print("  per-game changes (game: prog_lv/score -> llm_lv/score):")
    for g in sorted(p_per):
        if p_per[g] != l_per[g]:
            arrow = "UP" if l_per[g][1] > p_per[g][1] else "DOWN"
            print(f"    [{arrow}] {g}: {p_per[g][0]}/{p_per[g][1]:.3f} -> "
                  f"{l_per[g][0]}/{l_per[g][1]:.3f}")
    print("=" * 52)
finally:
    server.terminate()
    print("vLLM server stopped.")
