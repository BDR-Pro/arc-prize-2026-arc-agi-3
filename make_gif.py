"""Render the champion agent playing a game into an animated GIF.

Usage: python make_gif.py <game> [--agent my_agent.py] [--max-actions 4000]
                          [--out media/<game>.gif] [--scale 6]
"""
from __future__ import annotations

import argparse
import json
import glob
import logging
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
logging.disable(logging.CRITICAL)

from eval_harness import GAMES_DIR, FrameShim, _load_agent_class  # noqa: E402

# ARC palette (0-9 classic, 10-15 extended)
PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (87, 20, 87), (46, 26, 71),
    (255, 255, 255), (25, 60, 25), (100, 70, 30), (60, 60, 60),
]


def render(grid, scale):
    h, w = len(grid), len(grid[0])
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        row = grid[y]
        for x in range(w):
            px[x, y] = PALETTE[int(row[x]) & 0xF]
    return img.resize((w * scale, h * scale), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--agent", default=str(HERE / "my_agent.py"))
    ap.add_argument("--max-actions", type=int, default=4000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=400)
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

    latest = FrameShim(FrameDataRaw())
    frames_hist = []
    grids = []
    prev = None
    levels_at = []
    prev_lv = 0
    for i in range(args.max_actions):
        action = agent.choose_action(frames_hist, latest)
        data = None
        if action.is_complex():
            ad = getattr(action, "action_data", None)
            data = {"x": ad.x, "y": ad.y}
        raw = env.step(action, data=data)
        latest = FrameShim(raw)
        frames_hist.append(latest)
        frames_hist = frames_hist[-4:]
        if latest.frame:
            g = latest.frame[-1]
            if g != prev:                     # keep only changed frames
                grids.append([r[:] for r in g])
                prev = g
        if raw.levels_completed > prev_lv:
            levels_at.append((raw.levels_completed, len(grids)))
            prev_lv = raw.levels_completed
        if raw.state is GameState.WIN:
            break

    # subsample to max-frames, always keeping level-completion moments
    keep = set(range(0, len(grids), max(1, len(grids) // args.max_frames)))
    keep.update(idx - 1 for (_lv, idx) in levels_at)
    keep.add(len(grids) - 1)
    sel = [grids[i] for i in sorted(k for k in keep if 0 <= k < len(grids))]

    imgs = [render(g, args.scale) for g in sel]
    out = Path(args.out) if args.out else HERE / "media" / f"{args.game}.gif"
    out.parent.mkdir(exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=70, loop=0, optimize=True)
    print(f"{args.game}: levels={prev_lv} frames={len(sel)} -> {out} "
          f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
