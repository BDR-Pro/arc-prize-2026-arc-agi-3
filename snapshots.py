"""Save PNG snapshots of the agent playing a game, for visual inspection.

Usage: python snapshots.py <game> [--agent my_agent.py] [--every 250]
                           [--max-actions 3000] [--out <dir>]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
logging.disable(logging.CRITICAL)

from eval_harness import GAMES_DIR, FrameShim, _load_agent_class  # noqa: E402
from make_gif import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--agent", default=str(HERE / "my_agent.py"))
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--max-actions", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import arc_agi
    from arcengine import FrameDataRaw, GameState

    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                         environments_dir=str(GAMES_DIR))
    env = arc.make(args.game, seed=0)
    AgentCls = _load_agent_class(Path(args.agent))
    meta = json.load(open(glob.glob(
        str(GAMES_DIR / args.game / "*" / "metadata.json"))[0]))
    agent = AgentCls(game_id=meta["game_id"] + "#s2")

    out = Path(args.out) if args.out else HERE / "snaps" / args.game
    out.mkdir(parents=True, exist_ok=True)

    latest = FrameShim(FrameDataRaw())
    frames = []
    saved = 0
    for i in range(args.max_actions):
        action = agent.choose_action(frames, latest)
        data = None
        if action.is_complex():
            ad = getattr(action, "action_data", None)
            data = {"x": ad.x, "y": ad.y}
        raw = env.step(action, data=data)
        latest = FrameShim(raw)
        frames.append(latest)
        frames = frames[-4:]
        if latest.frame and (i % args.every == 0 or raw.levels_completed):
            if i % args.every == 0:
                img = render(latest.frame[-1], 8)
                img.save(out / f"a{i:04d}_lv{raw.levels_completed}.png")
                saved += 1
        if raw.state is GameState.WIN:
            break
    print(f"{args.game}: saved {saved} snapshots -> {out}")


if __name__ == "__main__":
    main()
