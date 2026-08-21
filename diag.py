"""Diagnostic runner: play one game in-process, dump agent internals.

Usage: python diag.py <game> [--agent my_agent.py] [--max-actions 4000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_harness import (FrameShim, GAMES_DIR, _load_agent_class,  # noqa: E402
                          find_game_meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--agent", default=str(HERE / "my_agent.py"))
    ap.add_argument("--max-actions", type=int, default=4000)
    args = ap.parse_args()

    import arc_agi
    from arcengine import GameState, FrameDataRaw

    meta = find_game_meta(args.game)
    full_id = meta.get("game_id", args.game)
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                         environments_dir=str(GAMES_DIR))
    env = arc.make(args.game, seed=0)
    AgentCls = _load_agent_class(Path(args.agent))
    agent = AgentCls(game_id=full_id)

    latest = FrameShim(FrameDataRaw())
    frames: list[FrameShim] = []
    actions = 0
    reasons: Counter = Counter()
    action_names: Counter = Counter()
    resets = 0
    levels_at: list[tuple[int, int]] = []
    prev_levels = 0

    while actions < args.max_actions and not agent.is_done(frames, latest):
        action = agent.choose_action(frames, latest)
        why = action.reasoning
        if isinstance(why, dict):
            why = why.get("why", "?")
        reasons[str(why).split(":")[0]] += 1
        action_names[action.name] += 1
        if action.name == "RESET":
            resets += 1
        data = None
        if action.is_complex():
            ad = getattr(action, "action_data", None)
            data = {"x": getattr(ad, "x", 0), "y": getattr(ad, "y", 0)}
        raw = env.step(action, data=data)
        actions += 1
        if raw is None:
            break
        latest = FrameShim(raw)
        frames.append(latest)
        if len(frames) > 4:
            frames.pop(0)
        if raw.levels_completed > prev_levels:
            levels_at.append((raw.levels_completed, actions))
            prev_levels = raw.levels_completed
        if raw.state is GameState.WIN:
            break

    avm = getattr(agent, "avm", None)
    out = {
        "game": args.game,
        "levels": prev_levels,
        "levels_at": levels_at,
        "actions": actions,
        "resets": resets,
        "n_states": len(agent.known_states),
        "n_score_up": len(agent.mem.model.score_up),
        "n_deadly": len(agent.mem.model.deadly),
        "n_good_clicks": len(agent.mem.good_clicks),
        "simple_seen": sorted(agent.mem.simple_seen),
        "complex_seen": agent.mem.complex_seen,
        "phase": agent.phase,
        "avatar_color": avm.avatar_color() if avm else None,
        "dirmap": {k: list(v) for k, v in (avm.direction_map() if avm else {}).items()},
        "goal_colors": sorted(getattr(agent, "goal_colors", [])),
        "reasons": dict(reasons.most_common()),
        "action_names": dict(action_names.most_common()),
    }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
