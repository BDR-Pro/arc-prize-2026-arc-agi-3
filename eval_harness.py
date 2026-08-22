"""Offline eval harness for ARC-AGI-3 agents against the cached public games.

Usage:
  python eval_harness.py --game ls20 [--agent my_agent.py]   # one game, prints JSON
  python eval_harness.py --all [--agent my_agent.py] [--jobs 8]

Mirrors the competition loop: agent.choose_action(frames, latest_frame) ->
env.step(action, data), 4000-action budget, per-level action counts, and the
official EnvironmentScoreCalculator (level-index-weighted, efficiency^2).
Frames are converted to lists-of-lists to match what Kaggle sends.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMES_DIR = HERE / "arc_games"
MAX_ACTIONS = 4000
PER_GAME_TIMEOUT = 3600  # seconds, driver-enforced (16k budgets under
#                            parallel contention need headroom)


def _stub_agents_module() -> None:
    """Provide agents.agent.Agent so my_agent.py imports cleanly."""
    if "agents.agent" in sys.modules:
        return
    pkg = types.ModuleType("agents")
    mod = types.ModuleType("agents.agent")

    class Agent:  # minimal stand-in for the starter framework base class
        MAX_ACTIONS = MAX_ACTIONS

        def __init__(self, game_id: str = "", **kwargs) -> None:
            self.game_id = game_id

        @property
        def name(self) -> str:
            return self.__class__.__name__

    mod.Agent = Agent
    pkg.agent = mod
    sys.modules["agents"] = pkg
    sys.modules["agents.agent"] = mod


def _load_agent_class(agent_path: Path):
    _stub_agents_module()
    spec = importlib.util.spec_from_file_location("my_agent_under_test", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


class FrameShim:
    """Kaggle-parity view of FrameDataRaw: lists-of-lists + score attr."""

    __slots__ = ("frame", "state", "available_actions", "levels_completed",
                 "score", "win_levels", "full_reset")

    def __init__(self, raw) -> None:
        self.frame = [g.tolist() for g in raw.frame]
        self.state = raw.state
        self.available_actions = list(raw.available_actions)
        self.levels_completed = raw.levels_completed
        self.score = raw.levels_completed
        self.win_levels = raw.win_levels
        self.full_reset = raw.full_reset


def find_game_meta(game: str) -> dict:
    gdir = GAMES_DIR / game
    for sub in sorted(gdir.iterdir()):
        meta = sub / "metadata.json"
        if meta.exists():
            return json.loads(meta.read_text())
    raise FileNotFoundError(f"no metadata.json under {gdir}")


def run_game(game: str, agent_path: Path, max_actions: int = MAX_ACTIONS,
             salt: int = 0) -> dict:
    from arcengine import GameState

    import arc_agi

    meta = find_game_meta(game)
    baseline = meta.get("baseline_actions") or []
    full_id = meta.get("game_id", game)

    arc = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(GAMES_DIR),
    )
    env = arc.make(game, seed=0)
    if env is None:
        return {"game": game, "error": "make() returned None"}

    AgentCls = _load_agent_class(agent_path)
    agent_gid = full_id if salt == 0 else f"{full_id}#s{salt}"
    agent = AgentCls(game_id=agent_gid)

    from arcengine import FrameDataRaw
    latest_raw = FrameDataRaw()  # NOT_PLAYED
    latest = FrameShim(latest_raw)
    frames: list[FrameShim] = []

    actions = 0
    prev_levels = 0
    level_start_action = 0
    per_level: list[tuple[int, int]] = []  # (level_index_1based, actions_taken)
    t0 = time.time()
    err = None

    try:
        while actions < max_actions and not agent.is_done(frames, latest)                 and actions < getattr(agent, "MAX_ACTIONS", 10**9):
            action = agent.choose_action(frames, latest)
            data = None
            if action.is_complex():
                ad = getattr(action, "action_data", None)
                data = {"x": getattr(ad, "x", 0), "y": getattr(ad, "y", 0)}
            raw = env.step(action, data=data)
            actions += 1
            if raw is None:
                err = "env.step returned None"
                break
            latest_raw = raw
            latest = FrameShim(raw)
            frames.append(latest)
            if len(frames) > 4:
                frames.pop(0)
            if raw.levels_completed > prev_levels:
                spent = actions - level_start_action
                for li in range(prev_levels + 1, raw.levels_completed + 1):
                    per_level.append((li, spent if li == prev_levels + 1 else 1))
                level_start_action = actions
                prev_levels = raw.levels_completed
            if raw.state is GameState.WIN:
                break
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        import traceback
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"

    elapsed = time.time() - t0

    # Official scoring
    calc = arc_agi.EnvironmentScoreCalculator(id=full_id)
    n_levels = latest_raw.win_levels or len(baseline) or max(prev_levels, 1)
    done_map = dict(per_level)
    for li in range(1, n_levels + 1):
        b = baseline[li - 1] if li - 1 < len(baseline) else 1
        if li in done_map:
            calc.add_level(li, True, done_map[li], b, game_id=full_id)
        else:
            calc.add_level(li, False, 0, b, game_id=full_id)
    score = calc.to_score(include_levels=False).score

    return {
        "game": game,
        "levels_completed": prev_levels,
        "win_levels": n_levels,
        "actions": actions,
        "per_level": per_level,
        "score": round(score, 4),
        "time_s": round(elapsed, 1),
        "error": err,
    }


def run_all(agent_path: Path, jobs: int, max_actions: int, salt: int = 0) -> None:
    games = sorted(d.name for d in GAMES_DIR.iterdir() if d.is_dir())
    results: dict[str, dict] = {}
    procs: dict[str, subprocess.Popen] = {}
    pending = list(games)
    starts: dict[str, float] = {}

    def launch(g: str) -> None:
        procs[g] = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--game", g,
             "--agent", str(agent_path), "--max-actions", str(max_actions),
             "--salt", str(salt)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        starts[g] = time.time()

    while pending or procs:
        while pending and len(procs) < jobs:
            launch(pending.pop(0))
        time.sleep(0.5)
        for g in list(procs):
            p = procs[g]
            if p.poll() is None:
                if time.time() - starts[g] > PER_GAME_TIMEOUT:
                    p.kill()
                    results[g] = {"game": g, "error": "timeout", "levels_completed": 0,
                                  "score": 0.0, "actions": max_actions}
                    del procs[g]
                continue
            out, errtxt = p.communicate()
            try:
                line = [l for l in out.splitlines() if l.startswith("{")][-1]
                results[g] = json.loads(line)
            except Exception:
                results[g] = {"game": g, "error": (errtxt or out)[-800:],
                              "levels_completed": 0, "score": 0.0, "actions": 0}
            del procs[g]
            r = results[g]
            print(f"  {g}: levels={r.get('levels_completed', 0)} "
                  f"score={r.get('score', 0.0):.4f} actions={r.get('actions', '?')} "
                  f"{'ERR: ' + str(r['error'])[:120] if r.get('error') else ''}",
                  flush=True)

    total_levels = sum(r.get("levels_completed", 0) for r in results.values())
    mean_score = sum(r.get("score", 0.0) for r in results.values()) / max(len(results), 1)
    print("\n=== SUMMARY ===")
    for g in games:
        r = results.get(g, {})
        print(f"{g:6s} levels={r.get('levels_completed', 0)}/{r.get('win_levels', '?')} "
              f"score={r.get('score', 0.0):8.4f} actions={r.get('actions', '?')}")
    print(f"\nTOTAL levels completed: {total_levels}")
    print(f"MEAN game score (0-100): {mean_score:.4f}")
    print(f"AGG (mean/100): {mean_score / 100:.7f}")
    out_path = HERE / "eval_results.json"
    out_path.write_text(json.dumps(
        {"agent": str(agent_path), "total_levels": total_levels,
         "mean_score": mean_score, "results": results}, indent=2))
    print(f"saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--agent", default=str(HERE / "my_agent.py"))
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    ap.add_argument("--salt", type=int, default=0)
    args = ap.parse_args()

    agent_path = Path(args.agent)
    if not agent_path.is_absolute():
        agent_path = HERE / agent_path

    if args.all:
        run_all(agent_path, args.jobs, args.max_actions, args.salt)
    elif args.game:
        print(json.dumps(run_game(args.game, agent_path, args.max_actions, args.salt)))
    else:
        ap.error("--game or --all required")


if __name__ == "__main__":
    main()
