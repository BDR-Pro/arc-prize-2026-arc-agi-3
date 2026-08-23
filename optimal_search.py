"""BFS over real engine states for the shortest level-1 solution.

Usage: python optimal_search.py <game> [--max-states 30000] [--ui-rows N]
Dedups states on the grid with the bottom N rows ignored (UI counters).
"""
import sys, copy, argparse, time
sys.path.insert(0, '.')
import logging
logging.disable(logging.CRITICAL)
from collections import deque
import numpy as np
from eval_harness import GAMES_DIR
import arc_agi
from arcengine import GameAction, ActionInput, GameState

ap = argparse.ArgumentParser()
ap.add_argument('game')
ap.add_argument('--max-states', type=int, default=30000)
ap.add_argument('--ui-rows', type=int, default=0)
ap.add_argument('--ui-rect', default='', help='r0,r1,c0,c1 region to ignore in dedup')
ap.add_argument('--clicks', action='store_true', help='also branch on object-centroid clicks')
a = ap.parse_args()

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(GAMES_DIR))
env = arc.make(a.game, seed=0)
f0 = env.step(GameAction.RESET)
avail = [GameAction.from_id(i) for i in f0.available_actions]
simple = [x for x in avail if not x.is_complex()]
print("available:", [x.name for x in avail])

def key(frame):
    g = frame.frame[-1]
    if a.ui_rows:
        g = g[:-a.ui_rows]
    if a.ui_rect:
        r0, r1, c0, c1 = map(int, a.ui_rect.split(','))
        g = g.astype(np.int16); g[r0:r1+1, c0:c1+1] = 255
    return g.tobytes()

def click_targets(frame):
    g = frame.frame[-1]
    vals, counts = np.unique(g, return_counts=True)
    bg = vals[np.argmax(counts)]
    from scipy import ndimage
    lab, n = ndimage.label(g != bg)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if 2 <= len(xs) <= 1500:
            out.append((int(xs.mean()), int(ys.mean())))
    out.sort(key=lambda t: (t[1], t[0]))
    return out[:16]

start = (env._game, [], f0)
seen = {key(f0)}
q = deque([start])
t0 = time.time()
explored = 0
while q:
    game, path, frame = q.popleft()
    explored += 1
    if explored > a.max_states:
        print("state cap hit; explored", explored); break
    branches = [(x, {}) for x in simple]
    if a.clicks and any(x.is_complex() for x in avail):
        for (cx, cy) in click_targets(frame):
            branches.append((GameAction.ACTION6, {"x": cx, "y": cy}))
    for act, data in branches:
        g2 = copy.deepcopy(game)
        fr = g2.perform_action(ActionInput(id=act, data=data), raw=True)
        if fr.state is GameState.GAME_OVER:
            continue
        if fr.levels_completed > 0:
            sol = path + [act.name + (f"({data['x']},{data['y']})" if data else "")]
            print(f"SOLUTION in {len(sol)} actions ({time.time()-t0:.0f}s, {explored} states):")
            print(" ".join(sol))
            sys.exit(0)
        k = key(fr)
        if k in seen:
            continue
        seen.add(k)
        q.append((g2, path + [act.name + (f"({data['x']},{data['y']})" if data else "")], fr))
print("no solution found; explored", explored, "states in", round(time.time()-t0), "s")
