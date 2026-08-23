"""ARC-AGI-3 agent (v16): world model + programmatic memory + planning + efficiency.

v16 — changes vs v15:
  - Early direction probing: the first actions of a game with simple
    actions cycle deterministically through each one 3x. The avatar
    model converges in ~12-24 actions instead of drifting there through
    random exploration, so navigation starts much earlier (efficiency:
    per-level scores are (baseline/actions)^2 — first-solve speed is
    everything).

v15 — changes vs v14 (from the 10-zero-games diagnosis):
  - Autonomous-motion filter: a color whose dominant delta is IDENTICAL for
    >=3 different actions is falling/scrolling on its own (sc25 gravity made
    all four actions map to "down" and navigation chased the falling block).
    Such colors can't be the avatar.
  - Adaptive volatility pressure: if >70% of the last 256 actions produced
    novel states, the mask thresholds tighten (0.20 -> 0.10 -> 0.05 ratio).
    re86/sk48/wa30 still exploded to 2400-3400 states under the base mask.
  - Inverse-action (undo) detection: an action that mostly REVERTS the
    previous transition is demoted in exploration/fallback (lf52's ACTION7
    was pressed 969x, undoing progress every time).
  - Novelty-biased fallback: prefer the action whose predicted next state
    has the lowest visit count (tr87 spent 3346 actions in uniform-random
    fallback inside a fully-explored 283-state graph).

v14 — changes vs v13:
  - Banked replays are pruned: actions whose RAW grid effect was nil are
    dropped before banking (a 3800-action discovery episode replays in a
    few hundred actions after death; skipping visible no-ops is safe and
    even saves hidden resources like energy).

v13 — changes vs v12:
  - Interface learning: per-action-name global effect/noop counters. An
    action that has NEVER changed the grid after 25 tries game-wide is
    dropped from exploration and fallback (vc33 diag: 2599/4000 actions
    were keyboard presses in a click-only game).
  - Useless-click suppression: if ACTION6 never changed the grid after 60
    tries game-wide, stop exploring clicks in keyboard games.
  - ROW/COLUMN volatility masking: a depleting energy bar changes a
    DIFFERENT cell each action, so per-cell masking never catches it
    (ls20/wa30/tr87 still had 2300-3900 states in v12). Rows/columns where
    any cell changes in >=40% of frames are masked wholesale.
  - Avatar model ignores masked cells: the depleting bar was outvoting the
    real avatar (ls20 "avatar" was a UI bar cell; dirmap stayed empty in
    every keyboard game).

v12 — changes vs v11:
  - Avatar detection rewritten: movers are found by diffing consecutive
    frames and matching color masses INSIDE the changed region's bounding
    box. v10/v11 required the avatar's color to form exactly one connected
    component in the whole grid — screen borders and UI bars sharing the
    color made detection fail in EVERY keyboard game (dirmap was always
    empty; navigation never ran).
  - locate() picks the connected component of avatar color nearest the
    last known position (was: centroid of all same-colored cells,
    including UI).

v11 — changes vs v10:
  - Volatility-masked state hashing: cells that change in >=20% of frames
    (counters, animations, energy bars) are excluded from the state hash.
    Fixes state explosion that made the world model useless and momentum
    fire forever in games with UI counters (e.g. ls20's energy bar).
  - Wall-COLOR learning: blocked moves teach wall colors; nav BFS stops
    pathing through walls after ~3 bumps instead of per-cell learning.
  - Walkable-color learning from successful avatar moves.
  - Nav arrival tries ALL untried non-movement actions, not just ACTION5.
  - Click-color productivity: clicks are prioritized toward colors whose
    clicks changed the grid before; provably-dead colors are skipped.
  - BFS planning uses a prebuilt adjacency index (was O(nodes*edges)).

v3 — changes vs v1 (the version that scored 0.00):
  - Action budget raised 400 -> 4000 (completing levels beats efficiency)
  - Progress replay keyed per level-start state (RESET is a LEVEL reset in
    competition mode; replaying level-1 moves at level-2 start was wasted)
  - available_actions entries normalized (real framework sends ints, not
    GameAction objects — v2 crashed on this in local play)

Architecture
------------
1. WORLD MODEL   TransitionModel learns (state, action) -> next state from
                 experience: which actions are no-ops, which change the grid,
                 which raise score, which kill.
2. MEMORY        GameMemory persists across resets within a game: the state
                 graph, per-state action stats, discovered click targets for
                 ACTION6, and the best known action sequence per level start
                 (replayed after death so progress is never re-earned).
3. PLANNING      BFS over the learned state graph toward (a) known
                 score-increasing transitions, else (b) the nearest frontier
                 state with untried actions. Plans are verified step-by-step
                 and abandoned on prediction mismatch.
4. EFFICIENCY    Known no-ops are skipped, visited states are not re-probed,
                 ACTION6 coordinates are sampled from object centroids and
                 recently-changed cells instead of uniformly from 64x64=4096
                 cells.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`; class named `MyAgent`.
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import random
import zlib
from collections import defaultdict, deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
MAX_ACTIONS_PER_GAME = 16000    # generous budget, wall-clock capped below
GAME_TIME_CAP_S = 160          # per-game wall-clock cap: 110 games must
#                                fit the Kaggle rerun window; when the cap
#                                hits, MAX_ACTIONS collapses to the count
#                                so the framework loop ends the game
DESPERATE_AFTER = 1200         # score==0 past this -> extended arsenal
NOOP_SKIP_THRESHOLD = 2        # times an action must no-op before we skip it
STUCK_WINDOW = 50              # actions with no new state before we force reset
MAX_COMPLEX_PER_STATE = 24
MOMENTUM_CAP = 60              # max consecutive repeats of one action
FIXATION_WINDOW = 200          # look-back for action-fixation detection
FIXATION_SHARE = 0.85          # one action dominating window w/o score-up
SUPPRESS_FOR = 120             # actions to ban a fixated action
RESEED_AFTER = 1000            # actions w/o score-up -> new exploration phase
CLICK_GRID_STRIDE = 8          # fallback coarse grid for ACTION6 exploration


class CellVolatility:
    """Tracks per-cell change frequency; masks hyper-volatile cells.

    Cells that change in >=RATIO of observed frames (step counters, energy
    bars, ambient animations) are excluded from the state hash. Without
    this, every action lands in a "new" state: the transition model learns
    nothing, no-op detection never fires, and momentum spams one key.
    The mask self-heals: a cell touched heavily early (e.g. a key sprite
    the agent spun in place) drops back out as frames accumulate.
    """

    RECOMPUTE_EVERY = 64
    ROW_RATIO = 0.40          # row/col changing this often is UI (bars)
    ROW_MIN = 16
    WARMUP_FRAMES = 32
    # (cell ratio, min changes) per pressure level; raised when the state
    # graph keeps exploding under the current mask
    PRESSURE_LEVELS = ((0.20, 12), (0.10, 8), (0.05, 5))

    def __init__(self) -> None:
        self.change: dict[tuple[int, int], int] = defaultdict(int)
        self.row_change: dict[int, int] = defaultdict(int)
        self.col_change: dict[int, int] = defaultdict(int)
        self.frames = 0
        self.h = 0
        self.w = 0
        self.prev: Optional[list[list[int]]] = None
        self.mask: frozenset[tuple[int, int]] = frozenset()
        self.mask_rev = 0
        self._since = 0
        self.pressure = 0

    def raise_pressure(self) -> None:
        if self.pressure < len(self.PRESSURE_LEVELS) - 1:
            self.pressure += 1
            self._recompute()

    def observe(self, grid: list[list[int]]) -> None:
        if not grid:
            return
        p = self.prev
        if p is not None and len(p) == len(grid) \
                and p and grid and len(p[0]) == len(grid[0]):
            self.h, self.w = len(grid), len(grid[0])
            rows_hit: set[int] = set()
            cols_hit: set[int] = set()
            for y, (rp, rc) in enumerate(zip(p, grid)):
                if rp != rc:
                    rows_hit.add(y)
                    for x, (a, b) in enumerate(zip(rp, rc)):
                        if a != b:
                            self.change[(x, y)] += 1
                            cols_hit.add(x)
            for y in rows_hit:
                self.row_change[y] += 1
            for x in cols_hit:
                self.col_change[x] += 1
            self.frames += 1
        self.prev = [list(row) for row in grid]
        self._since += 1
        if self._since >= self.RECOMPUTE_EVERY:
            self._since = 0
            self._recompute()

    def _recompute(self) -> None:
        if self.frames < self.WARMUP_FRAMES:
            return
        ratio, min_ch = self.PRESSURE_LEVELS[self.pressure]
        thr = max(min_ch, int(ratio * self.frames))
        cells = {c for c, n in self.change.items() if n >= thr}
        # whole-row / whole-column UI masking (depleting bars change a
        # different cell each frame; per-cell counts never trip)
        row_thr = max(self.ROW_MIN, int(self.ROW_RATIO * self.frames))
        for y, n in self.row_change.items():
            if n >= row_thr:
                cells.update((x, y) for x in range(self.w))
        for x, n in self.col_change.items():
            if n >= row_thr:
                cells.update((x, y) for y in range(self.h))
        m = frozenset(cells)
        if m != self.mask:
            self.mask = m
            self.mask_rev += 1

    def hash_grid(self, grid: list[list[int]]) -> int:
        if not grid:
            return 0
        mask = self.mask
        h = zlib.crc32(bytes((self.mask_rev & 0xFF,)))
        if mask:
            for y, row in enumerate(grid):
                h = zlib.crc32(bytes(
                    0xFF if (x, y) in mask else (v & 0xFF)
                    for x, v in enumerate(row)), h)
        else:
            for row in grid:
                h = zlib.crc32(bytes(v & 0xFF for v in row), h)
        return h


def _grid_hash(frame: FrameData) -> int:
    """Unmasked fallback hash (used only before memory exists)."""
    grids = getattr(frame, "frame", None) or []
    if not grids:
        return 0
    return zlib.crc32(b"".join(
        bytes(v & 0xFF for v in row) for row in grids[-1]))


def _last_layer(frame: FrameData) -> list[list[int]]:
    grids = getattr(frame, "frame", None) or []
    return grids[-1] if grids else []


def _diff_cells(a: list[list[int]], b: list[list[int]]) -> list[tuple[int, int]]:
    """Cells (x, y) that differ between two grids of the same shape."""
    out: list[tuple[int, int]] = []
    for y, (ra, rb) in enumerate(zip(a, b)):
        for x, (va, vb) in enumerate(zip(ra, rb)):
            if va != vb:
                out.append((x, y))
    return out


def _object_centroids(grid: list[list[int]]) -> list[tuple[int, int]]:
    """Centroids of connected same-color components (non-background).

    Background is taken as the most frequent color. These are the natural
    candidate targets for ACTION6 clicks.
    """
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    counts: dict[int, int] = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    background = max(counts, key=counts.get)  # type: ignore[arg-type]

    seen = [[False] * w for _ in range(h)]
    centroids: list[tuple[int, int]] = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or grid[y0][x0] == background:
                continue
            color = grid[y0][x0]
            stack = [(x0, y0)]
            seen[y0][x0] = True
            xs, ys, n = 0, 0, 0
            while stack:
                x, y = stack.pop()
                xs, ys, n = xs + x, ys + y, n + 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] \
                            and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if n:
                centroids.append((xs // n, ys // n))
    return centroids


def _enclosed_cells(grid: list[list[int]]) -> list[tuple[int, int]]:
    """Interior points of background pockets NOT reachable from the border.

    Catches ring/donut shapes where the object centroid falls in the hole --
    a common click target in ARC-style games.
    """
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    counts: dict[int, int] = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    background = max(counts, key=counts.get)  # type: ignore[arg-type]

    reachable = [[False] * w for _ in range(h)]
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if grid[y][x] == background and not reachable[y][x]:
                reachable[y][x] = True
                stack.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if grid[y][x] == background and not reachable[y][x]:
                reachable[y][x] = True
                stack.append((x, y))
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not reachable[ny][nx] \
                    and grid[ny][nx] == background:
                reachable[ny][nx] = True
                stack.append((nx, ny))

    out: list[tuple[int, int]] = []
    seen = [[False] * w for _ in range(h)]
    for y0 in range(h):
        for x0 in range(w):
            if grid[y0][x0] != background or reachable[y0][x0] or seen[y0][x0]:
                continue
            comp = [(x0, y0)]
            seen[y0][x0] = True
            xs, ys, n = 0, 0, 0
            while comp:
                x, y = comp.pop()
                xs, ys, n = xs + x, ys + y, n + 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] \
                            and grid[ny][nx] == background \
                            and not reachable[ny][nx]:
                        seen[ny][nx] = True
                        comp.append((nx, ny))
            if n:
                out.append((xs // n, ys // n))
    return out


def _symmetry_break_cells(grid: list[list[int]]) -> list[tuple[int, int]]:
    """Cells breaking a near-perfect mirror symmetry (>=85% match).

    A common ARC-3 click-puzzle archetype: restore the symmetric picture.
    Returns both sides of each mismatched pair, nearest-to-complete axis
    first; empty when the grid is not close to symmetric.
    """
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    results: list[tuple[float, list[tuple[int, int]]]] = []
    # horizontal mirror (left-right)
    same = 0
    diffs_h: list[tuple[int, int]] = []
    for y in range(h):
        row = grid[y]
        for x in range(w // 2):
            if row[x] == row[w - 1 - x]:
                same += 1
            else:
                diffs_h.append((x, y))
                diffs_h.append((w - 1 - x, y))
    total = same + len(diffs_h) // 2
    if diffs_h and total and len(diffs_h) // 2 <= 8:
        results.append((len(diffs_h) / total, diffs_h))
    # vertical mirror (top-bottom)
    same = 0
    diffs_v: list[tuple[int, int]] = []
    for y in range(h // 2):
        ra, rb = grid[y], grid[h - 1 - y]
        for x in range(w):
            if ra[x] == rb[x]:
                same += 1
            else:
                diffs_v.append((x, y))
                diffs_v.append((x, h - 1 - y))
    total = same + len(diffs_v) // 2
    if diffs_v and total and len(diffs_v) // 2 <= 8:
        results.append((len(diffs_v) / total, diffs_v))
    results.sort(key=lambda r: r[0])
    out: list[tuple[int, int]] = []
    for _ratio, cells in results:
        out.extend(cells)
    return out


def _template_mismatch_cells(grid: list[list[int]]
                             ) -> list[tuple[int, int]]:
    """Copy-task detector: two same-size non-background regions matching
    >=60% cell-wise are reference+working copies; the mismatching cells
    (in both, we cannot know which is editable) are prime click targets.
    Conservative: fires only for ONE strong pair with <=40 mismatches."""
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    counts: dict[int, int] = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    background = max(counts, key=counts.get)  # type: ignore[arg-type]

    # 4-connected components of the non-background MASK (multi-color
    # patterns count as one region)
    seen = [[False] * w for _ in range(h)]
    regions: list[tuple[int, int, int, int, int]] = []  # x0,y0,x1,y1,n
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or grid[y0][x0] == background:
                continue
            stack = [(x0, y0)]
            seen[y0][x0] = True
            minx = maxx = x0
            miny = maxy = y0
            n = 0
            while stack:
                x, y = stack.pop()
                n += 1
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] \
                            and grid[ny][nx] != background:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if n >= 12:
                regions.append((minx, miny, maxx, maxy, n))

    best: Optional[tuple[float, list[tuple[int, int]]]] = None
    for i in range(len(regions)):
        ax0, ay0, ax1, ay1, _an = regions[i]
        aw, ah = ax1 - ax0 + 1, ay1 - ay0 + 1
        if aw * ah > 1200:
            continue
        for j in range(i + 1, len(regions)):
            bx0, by0, bx1, by1, _bn = regions[j]
            if (bx1 - bx0 + 1, by1 - by0 + 1) != (aw, ah):
                continue
            same = 0
            mism: list[tuple[int, int]] = []
            for dy in range(ah):
                for dx in range(aw):
                    va = grid[ay0 + dy][ax0 + dx]
                    vb = grid[by0 + dy][bx0 + dx]
                    if va == vb:
                        same += 1
                    else:
                        mism.append((ax0 + dx, ay0 + dy))
                        mism.append((bx0 + dx, by0 + dy))
            total = aw * ah
            frac = same / total
            if frac >= 0.6 and 0 < len(mism) // 2 <= 40:
                if best is None or frac > best[0]:
                    best = (frac, mism)
    return best[1] if best else []


def _components_info(grid: list[list[int]]
                     ) -> list[tuple[int, int, int, int]]:
    """(cx, cy, size, color) per connected non-background component."""
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    counts: dict[int, int] = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    background = max(counts, key=counts.get)  # type: ignore[arg-type]
    seen = [[False] * w for _ in range(h)]
    out: list[tuple[int, int, int, int]] = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or grid[y0][x0] == background:
                continue
            color = grid[y0][x0]
            stack = [(x0, y0)]
            seen[y0][x0] = True
            xs, ys, n = 0, 0, 0
            while stack:
                x, y = stack.pop()
                xs, ys, n = xs + x, ys + y, n + 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] \
                            and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if n:
                out.append((xs // n, ys // n, n, color))
    return out


def _color_counts(grid: list[list[int]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    return counts


class AvatarModel:
    """Object-centric model: find the avatar and learn what actions do.

    Between consecutive frames, objects that moved are matched by
    (color, size). Each mover votes (color, action) -> delta. The avatar is
    the color whose deltas are most consistent per action; enemies moving on
    their own schedule spread their votes across deltas and lose.
    """

    MIN_OBS = 3          # observations before a direction is trusted
    CONSISTENCY = 0.7    # dominant delta must be >=70% of votes

    def __init__(self) -> None:
        self.stats: dict[tuple[int, str], dict[tuple[int, int], int]] = \
            defaultdict(lambda: defaultdict(int))
        self.pos: Optional[tuple[int, int]] = None
        self.size: Optional[int] = None
        self.blocked_tries: dict[tuple[int, int], int] = defaultdict(int)
        # rule induction: which COLORS block movement, which are walkable
        self.blocked_colors: dict[int, int] = defaultdict(int)
        self.entered_colors: dict[int, int] = defaultdict(int)

    MAX_CHANGED = 220     # bigger diffs are scene transitions, not movement
    MAX_MOVER_SIZE = 300  # colors covering more cells can't be the avatar

    def update(self, prev_grid: list[list[int]], cur_grid: list[list[int]],
               action_name: str,
               ignore: frozenset[tuple[int, int]] = frozenset()) -> None:
        if not prev_grid or not cur_grid or len(prev_grid) != len(cur_grid) \
                or len(prev_grid[0]) != len(cur_grid[0]):
            return
        h, w = len(prev_grid), len(prev_grid[0])
        changed = [(x, y) for y in range(h) for x in range(w)
                   if prev_grid[y][x] != cur_grid[y][x]
                   and (x, y) not in ignore]
        moved_any = False
        if changed and len(changed) <= self.MAX_CHANGED:
            xs = [c[0] for c in changed]
            ys = [c[1] for c in changed]
            x0, x1 = max(0, min(xs) - 2), min(w - 1, max(xs) + 2)
            y0, y1 = max(0, min(ys) - 2), min(h - 1, max(ys) + 2)
            global_counts = _color_counts(prev_grid)
            colors = set()
            for (x, y) in changed:
                colors.add(prev_grid[y][x])
                colors.add(cur_grid[y][x])
            av_color = self.avatar_color()
            for color in colors:
                # large masses (background, floors) anti-move when the
                # avatar moves; they can't be the avatar
                if global_counts.get(color, 0) > self.MAX_MOVER_SIZE:
                    continue
                pc = [(x, y) for y in range(y0, y1 + 1)
                      for x in range(x0, x1 + 1) if prev_grid[y][x] == color]
                cc = [(x, y) for y in range(y0, y1 + 1)
                      for x in range(x0, x1 + 1) if cur_grid[y][x] == color]
                if not pc or not cc:
                    continue
                if abs(len(pc) - len(cc)) > max(2, len(pc) // 2):
                    continue
                dx = round(sum(c[0] for c in cc) / len(cc)
                           - sum(c[0] for c in pc) / len(pc))
                dy = round(sum(c[1] for c in cc) / len(cc)
                           - sum(c[1] for c in pc) / len(pc))
                if (dx == 0 and dy == 0) or max(abs(dx), abs(dy)) > 3:
                    continue
                self.stats[(color, action_name)][(dx, dy)] += 1
                moved_any = True
                if color == av_color and max(abs(dx), abs(dy)) <= 3:
                    self.size = len(cc)
                    # cells newly covered by the avatar were walkable
                    gained = set(cc) - set(pc)
                    for (gx, gy) in gained:
                        self.entered_colors[prev_grid[gy][gx]] += 1
        if not moved_any and self.pos is not None:
            dirs = self.direction_map()
            d = dirs.get(action_name)
            if d is not None:
                # blame every cell along the attempted step (with 2-cell
                # steps the wall is usually the INTERMEDIATE cell)
                n = max(abs(d[0]), abs(d[1]))
                for i in range(1, n + 1):
                    cx = self.pos[0] + round(d[0] * i / n)
                    cy = self.pos[1] + round(d[1] * i / n)
                    if 0 <= cy < len(prev_grid) and 0 <= cx < len(prev_grid[0]):
                        self.blocked_tries[(cx, cy)] += 1
                        self.blocked_colors[prev_grid[cy][cx]] += 1

    def wall_colors(self) -> set[int]:
        """Colors that block movement (>=3 bumps, never walked onto).

        Colors that have ever MOVED under our actions are excluded: a
        pushable block bumps like a wall but is the opposite of one
        (Sokoban-style games need the avatar to keep walking into it)."""
        movers = {c for (c, _a), deltas in self.stats.items()
                  if sum(deltas.values()) >= 2}
        return {c for c, n in self.blocked_colors.items()
                if n >= 3 and self.entered_colors.get(c, 0) == 0
                and c not in movers}

    def _autonomous(self, color: int) -> bool:
        """Same dominant delta for >=3 actions = gravity/conveyor, not
        player control (sc25: falling pieces made every action vote
        'down')."""
        tops: list[tuple[int, int]] = []
        for (c, _act), deltas in self.stats.items():
            if c != color:
                continue
            total = sum(deltas.values())
            if total < 2:
                continue
            tops.append(max(deltas.items(), key=lambda kv: kv[1])[0])
        return len(tops) >= 3 and len(set(tops)) == 1

    def avatar_color(self) -> Optional[int]:
        best_color, best_score = None, 0
        by_color: dict[int, int] = defaultdict(int)
        for (color, _act), deltas in self.stats.items():
            total = sum(deltas.values())
            top = max(deltas.values())
            if total >= self.MIN_OBS and top / total >= self.CONSISTENCY:
                by_color[color] += top
        for color, score in by_color.items():
            if score > best_score and not self._autonomous(color):
                best_color, best_score = color, score
        return best_color if best_score >= self.MIN_OBS else None

    def direction_map(self) -> dict[str, tuple[int, int]]:
        color = self.avatar_color()
        if color is None:
            return {}
        out: dict[str, tuple[int, int]] = {}
        for (c, act), deltas in self.stats.items():
            if c != color:
                continue
            total = sum(deltas.values())
            delta, top = max(deltas.items(), key=lambda kv: kv[1])
            # accept 1-3 cell steps: ls20's avatar moves 2 cells per press
            # and the ==1 filter kept navigation dead in every such game
            if total >= self.MIN_OBS and top / total >= self.CONSISTENCY \
                    and 1 <= max(abs(delta[0]), abs(delta[1])) <= 3:
                out[act] = delta
        # two keys never share a delta in a real control scheme: shared
        # deltas are ambient motion (gravity) that leaked past the
        # autonomy filter (sc25 had two actions both mapping to "down")
        by_delta: dict[tuple[int, int], list[str]] = defaultdict(list)
        for act, d in out.items():
            by_delta[d].append(act)
        for d, acts in by_delta.items():
            if len(acts) > 1:
                for a in acts:
                    del out[a]
        return out

    def locate(self, grid: list[list[int]]) -> Optional[tuple[int, int]]:
        """Centroid of the avatar-colored component nearest the last known
        position (UI elements sharing the color are separate components)."""
        color = self.avatar_color()
        if color is None or not grid:
            return None
        h, w = len(grid), len(grid[0])
        comps: list[list[tuple[int, int]]] = []
        seen: set[tuple[int, int]] = set()
        for y in range(h):
            for x in range(w):
                if grid[y][x] != color or (x, y) in seen:
                    continue
                stack = [(x, y)]
                seen.add((x, y))
                cells: list[tuple[int, int]] = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h \
                                and (nx, ny) not in seen \
                                and grid[ny][nx] == color:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                comps.append(cells)
        if not comps:
            return None
        if self.pos is not None:
            px, py = self.pos
            comps.sort(key=lambda cs: min(
                abs(c[0] - px) + abs(c[1] - py) for c in cs))
        elif self.size is not None:
            comps.sort(key=lambda cs: abs(len(cs) - self.size))
        else:
            comps.sort(key=len)
        cells = comps[0]
        cx = sum(c[0] for c in cells) // len(cells)
        cy = sum(c[1] for c in cells) // len(cells)
        self.pos = (cx, cy)
        return self.pos


class ActionKey:
    """Hashable key for an action incl. ACTION6 coordinates."""

    __slots__ = ("name", "x", "y")

    _ORDER = {"RESET": 0, "ACTION1": 1, "ACTION2": 2, "ACTION3": 3,
              "ACTION4": 4, "ACTION5": 5, "ACTION6": 6, "ACTION7": 7}

    def __init__(self, action: GameAction, x: int = -1, y: int = -1) -> None:
        self.name = action.name
        self.x = x
        self.y = y

    def __hash__(self) -> int:
        # deterministic across processes (no str hashing)
        return hash((self._ORDER.get(self.name, 99), self.x, self.y))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ActionKey) and \
            (self.name, self.x, self.y) == (other.name, other.x, other.y)

    def __repr__(self) -> str:
        if self.x >= 0:
            return f"{self.name}({self.x},{self.y})"
        return self.name


class TransitionModel:
    """Learned world model: (state, action) -> outcome statistics."""

    def __init__(self) -> None:
        # (state, akey) -> {next_state: count}
        self.next: dict[tuple[int, ActionKey], dict[int, int]] = \
            defaultdict(lambda: defaultdict(int))
        self.noops: dict[tuple[int, ActionKey], int] = defaultdict(int)
        self.score_up: dict[tuple[int, ActionKey], None] = {}
        self.deadly: set[tuple[int, ActionKey]] = set()

    def observe(self, s: int, akey: ActionKey, s2: int,
                dscore: int, died: bool) -> None:
        self.next[(s, akey)][s2] += 1
        if s2 == s:
            self.noops[(s, akey)] += 1
        if dscore > 0:
            self.score_up[(s, akey)] = None
        if died:
            self.deadly.add((s, akey))

    def is_known_noop(self, s: int, akey: ActionKey) -> bool:
        key = (s, akey)
        return self.noops[key] >= NOOP_SKIP_THRESHOLD and \
            len(self.next[key]) <= 1

    def predicted_next(self, s: int, akey: ActionKey) -> Optional[int]:
        outcomes = self.next.get((s, akey))
        if not outcomes:
            return None
        return max(outcomes, key=outcomes.get)  # type: ignore[arg-type]


class GameMemory:
    """Programmatic memory for one game, persistent across resets."""

    def __init__(self) -> None:
        self.model = TransitionModel()
        # state -> ActionKeys already tried there
        self.tried: dict[int, set[ActionKey]] = defaultdict(set)
        self.visit_count: dict[int, int] = defaultdict(int)
        # best action sequence per starting state that achieved score-ups.
        # Keyed by the post-reset state hash: in competition mode RESET is a
        # LEVEL reset, so after completing level 1 a death restarts at level
        # 2's start state — a level-1 sequence would be wrong there.
        self.best_prefix_by_start: dict[int, tuple[int, list[ActionKey]]] = {}
        # ACTION6 targets that provably changed the grid somewhere
        self.good_clicks: set[tuple[int, int]] = set()
        # action names this game has ever offered (RESET excluded)
        self.simple_seen: set[str] = set()
        self.complex_seen = False
        # volatility-masked state identity
        self.vol = CellVolatility()
        # click rule induction: which colors respond to clicks
        self.click_color_good: dict[int, int] = defaultdict(int)
        # contextual dead-click rules: "(x, y) showing color c is dead".
        # Keyed by appearance so an armed button (changed color) escapes
        # the ban — plain per-cell banning cost 3 levels (v43/v44).
        self.click_ctx_good: dict[tuple[int, int, int], int] = \
            defaultdict(int)
        self.click_ctx_bad: dict[tuple[int, int, int], int] = \
            defaultdict(int)
        self.click_color_bad: dict[int, int] = defaultdict(int)
        # interface learning: which action NAMES ever do anything
        self.action_effect: dict[str, int] = defaultdict(int)
        self.action_noop: dict[str, int] = defaultdict(int)
        # undo detection: action mostly reverts the previous transition
        self.action_revert: dict[str, int] = defaultdict(int)
        self.action_moves: dict[str, int] = defaultdict(int)


class MyAgent(Agent):
    """World-model agent: explore -> model -> plan -> act efficiently."""

    MAX_ACTIONS = MAX_ACTIONS_PER_GAME

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rng = random.Random(zlib.crc32(str(self.game_id).encode()))
        self.mem = GameMemory()
        # episode-local
        self.prev_state: Optional[int] = None
        self.prev_action: Optional[ActionKey] = None
        self.prev_score = 0
        self.prev_grid: list[list[int]] = []
        self.episode_actions: list[ActionKey] = []
        self.episode_effects: list[bool] = []  # raw grid changed?
        self.episode_start_state: Optional[int] = None
        self.plan: deque[ActionKey] = deque()
        self.plan_expected: deque[int] = deque()
        self.replay_queue: deque[ActionKey] = deque()
        self.since_new_state = 0
        self.known_states: set[int] = set()
        self.action_count = 0
        self.momentum: Optional[ActionKey] = None
        self.momentum_streak = 0
        self.recent_actions: deque[str] = deque(maxlen=FIXATION_WINDOW)
        self.suppress_until: dict[str, int] = {}
        self.last_scoreup_at = 0
        self.phase = 0
        # state-explosion monitor (drives volatility pressure)
        self._novel_recent = 0
        self._window_start = 0
        # last observed transition, for undo detection
        self._last_transition: Optional[tuple[int, int]] = None
        # deterministic opening probe (built on the first playable frame)
        self._probe_queue: Optional[deque[str]] = None
        # telemetry
        self._reason_counts: dict[str, int] = {}
        self._last_s2: Optional[int] = None
        self._cur_counts: Optional[dict[int, int]] = None
        # cross-level transfer state
        self._won_seq: list = []
        self._won_clicks: list = []
        self._won_seq_tried = True
        self._strategy_queue: deque = deque()
        self._won_click_colors: set = set()
        self._finisher_pending = False
        # solver mode state
        self._solver: dict = {}
        self._last_solver_pos: Optional[tuple[int, int]] = None
        # reaction flag (desperation-mode interact trigger)
        self._world_reacted = False
        # object-centric navigation
        self.avm = AvatarModel()
        self.visited_cells: set[tuple[int, int]] = set()
        self.exhausted_targets: set[tuple[int, int]] = set()
        self.goal_colors: set[int] = set()
        self.nav_target: Optional[tuple[int, int]] = None

    @property
    def name(self) -> str:
        return f"{super().name}.worldmodel"

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData],
                      latest_frame: FrameData) -> GameAction:
        self.action_count += 1
        if self.action_count % 500 == 0:
            self._telemetry("tick")
        # wall-clock budget: collapse MAX_ACTIONS when the per-game time
        # cap is reached so the framework loop moves on to the next game
        import time as _t
        if not hasattr(self, "_t0"):
            self._t0 = _t.time()
        elif _t.time() - self._t0 > GAME_TIME_CAP_S                 and self.action_count >= 4000                 and self.action_count < self.MAX_ACTIONS:
            # the 4000-action floor is guaranteed regardless of time
            # (v54 proved 110x4000 fits the rerun window); the wall cap
            # only trims the EXTRA budget in slow games
            # set the ceiling BELOW the current count, once — setting it
            # equal kept the loop alive forever (counter is one behind)
            self.MAX_ACTIONS = max(0, self.action_count - 2)

        # -- Reset handling ------------------------------------------------
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._learn(latest_frame, died=latest_frame.state is GameState.GAME_OVER)
            self._start_episode()
            self._rc("reset")
            action = GameAction.RESET
            action.reasoning = "reset: start/retry episode"
            return action

        # -- Learn from the outcome of the previous action -----------------
        # volatility is observed FIRST so _learn and the s below hash the
        # frame under the same mask revision
        grid = _last_layer(latest_frame)
        self.mem.vol.observe(grid)
        self._learn(latest_frame, died=False)

        s = self._last_s2 if self._last_s2 is not None else self._hash(grid)
        self._cur_grid = grid
        self._cur_counts = None
        available = self._available_actions(latest_frame)
        self._record_available(available)
        self._break_fixation()

        # -- Phase reseed: deterministic, but escape long no-progress ruts --
        if self.action_count - self.last_scoreup_at >= \
                RESEED_AFTER * (self.phase + 1):
            self.phase += 1
            self.rng.seed(zlib.crc32(str(self.game_id).encode()) + self.phase)
            self.suppress_until.clear()
            self.momentum = None
            self.momentum_streak = 0
            self.plan.clear()
            self.plan_expected.clear()
            # allow periodic retesting of "useless" actions and dead rules
            for k in list(self.mem.action_noop):
                self.mem.action_noop[k] //= 2
            for k in list(self.mem.click_ctx_bad):
                self.mem.click_ctx_bad[k] //= 2

        # -- State-explosion monitor: tighten the volatility mask ----------
        # Gated to games that never scored: pressure changes state identity
        # wholesale, which wrecked productive games' learned graphs in v15
        # (lp85 5->4, tu93 3->2 levels).
        if self.action_count - self._window_start >= 256:
            window = self.action_count - self._window_start
            if self.action_count >= 512 and self.last_scoreup_at == 0 \
                    and self._novel_recent > 0.85 * window:
                self.mem.vol.raise_pressure()
            self._novel_recent = 0
            self._window_start = self.action_count

        # -- First frame of an episode: bind start state, load matching replay
        if self.episode_start_state is None:
            self.episode_start_state = s
            banked = self.mem.best_prefix_by_start.get(s)
            if banked:
                self.replay_queue = deque(banked[1])

        # -- 1. Replay known-good prefix after a reset ----------------------
        if self.replay_queue:
            akey = self.replay_queue.popleft()
            if self._is_legal(akey, available):
                return self._emit(akey, s, "replay: known-good prefix")
            self.replay_queue.clear()

        # -- 1.1 STRATEGY REPLAY: one-shot attempt of the PREVIOUS
        # level's winning sequence at the new level (levels in a game
        # share structure; the lp85 cascade shows this is worth 8-14
        # points when it lands)
        if getattr(self, "_won_seq", None) and not self._won_seq_tried \
                and self.prev_score > 0 \
                and self.action_count - self.last_scoreup_at < 10:
            self._won_seq_tried = True
            self._strategy_queue = deque(self._won_seq)
            self._finisher_pending = len(self._won_seq) > 20
        sq = getattr(self, "_strategy_queue", None)
        if sq:
            akey = sq.popleft()
            if self._is_legal(akey, available) \
                    and (s, akey) not in self.mem.model.deadly:
                self.since_new_state = 0
                return self._emit(akey, s, "strategy: replay last win")
            sq.clear()
        elif getattr(self, "_finisher_pending", False) \
                and self.prev_score > 0 \
                and 150 < self.action_count - self.last_scoreup_at < 400:
            # second stage: the FINISHING MOVE alone — the last actions
            # of the win are the actual solution and often transfer even
            # when the wandering prefix does not
            self._finisher_pending = False
            self._strategy_queue = deque(self._won_seq[-20:])

        # -- 1.2 SOLVER MODE: when a known archetype is detected with
        # confidence and the game is stagnant, exit the exploration stack
        # and execute the archetype's dedicated policy at scripted pace.
        # Exploration solves levels at 10-50x human baseline = ~1% credit;
        # only direct execution can approach baseline speed.
        sv = self._solver_action(s, grid, available)
        if sv is not None:
            return self._emit(sv, s, "solver: archetype execution")

        # -- 1.5 Opening probe: press each simple action 3x to learn the
        # interface (avatar + directions) as fast as possible --------------
        if self._probe_queue is None and self.action_count <= 3:
            names = sorted({a.name for a in available
                            if a is not GameAction.RESET
                            and not a.is_complex()})
            self._probe_queue = deque(n for n in names for _ in range(3))
        if self._probe_queue:
            while self._probe_queue:
                nm = self._probe_queue.popleft()
                akey = ActionKey(GameAction[nm])
                if self._is_legal(akey, available) \
                        and (s, akey) not in self.mem.model.deadly:
                    return self._emit(akey, s, "probe: learn interface")

        # -- 2. Follow current plan (verified step-by-step) -----------------
        if self.plan:
            expected = self.plan_expected.popleft() if self.plan_expected else None
            if expected is not None and expected != -1 \
                    and self.prev_state is not None and expected != s:
                # world diverged from prediction: drop the plan
                self.plan.clear()
                self.plan_expected.clear()
            else:
                akey = self.plan.popleft()
                if self._is_legal(akey, available):
                    return self._emit(akey, s, "plan: executing path")
                self.plan.clear()
                self.plan_expected.clear()

        # -- 3. Exploit: known score-up transition from this state ----------
        for (ms, akey) in self.mem.model.score_up:
            if ms == s and self._is_legal(akey, available) \
                    and (s, akey) not in self.mem.model.deadly:
                return self._emit(akey, s, "exploit: known score-up")

        # -- 3.1 React (desperation only): our last move lit something up
        # near the avatar — try interaction keys while it is still lit
        if self._world_reacted and self._desperate():
            move_names = set(self.avm.direction_map())
            for a in available:
                if a is GameAction.RESET or a.is_complex() \
                        or a.name in move_names:
                    continue
                ak = ActionKey(a)
                if ak not in self.mem.tried[s] \
                        and (s, ak) not in self.mem.model.deadly:
                    return self._emit(ak, s, "react: world lit up")

        # -- 3.2 Navigate: avatar known + directions confirmed --------------
        nav = self._nav_action(s, grid, available)
        if nav is not None:
            return self._emit(nav, s, "navigate: toward target")

        # -- 3.5 Momentum: last action kept producing new states ------------
        if self.momentum is not None and self.momentum_streak < MOMENTUM_CAP:
            akey = self.momentum
            if self._is_legal(akey, available) \
                    and not self._suppressed(akey.name) \
                    and not self.mem.model.is_known_noop(s, akey) \
                    and (s, akey) not in self.mem.model.deadly:
                self.momentum_streak += 1
                return self._emit(akey, s, "momentum: novelty streak")
            self.momentum = None
        if self.momentum_streak >= MOMENTUM_CAP:
            self.momentum = None
            self.momentum_streak = 0

        # -- 4. Explore locally: untried, non-noop, non-deadly action here --
        akey = self._pick_exploration_action(s, grid, available)
        if akey is not None:
            return self._emit(akey, s, "explore: novel action")

        # -- 5. Plan a path to a score-up or frontier state -----------------
        plan = self._bfs_plan(s, available)
        if plan:
            self.plan = deque(plan)
            # precompute expected states along the plan for verification
            self.plan_expected = deque()
            cur = s
            for akey in plan:
                nxt = self.mem.model.predicted_next(cur, akey)
                self.plan_expected.append(nxt if nxt is not None else -1)
                cur = nxt if nxt is not None else cur
            akey = self.plan.popleft()
            if self.plan_expected:
                self.plan_expected.popleft()
            return self._emit(akey, s, "plan: toward goal/frontier")

        # -- 6. Nothing new reachable: reset if stuck, else random legal ----
        if self.since_new_state >= STUCK_WINDOW and self._can_reset(latest_frame):
            self._start_episode()
            action = GameAction.RESET
            action.reasoning = "reset: stuck, no new states"
            return action
        akey = self._fallback_action(s, available)
        return self._emit(akey, s, "fallback: random legal action")

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _hash(self, grid: list[list[int]]) -> int:
        return self.mem.vol.hash_grid(grid)

    def _learn(self, latest_frame: FrameData, died: bool) -> None:
        self._last_s2 = None
        if self.prev_state is None or self.prev_action is None:
            return
        s2 = self._hash(_last_layer(latest_frame))
        self._last_s2 = s2  # choose_action reuses this for the same frame
        # raw (unmasked) effect of the consumed action, for replay pruning
        _cur = _last_layer(latest_frame)
        self.episode_effects.append((not self.prev_grid) or
                                    self.prev_grid != _cur)
        score = getattr(latest_frame, "score", None)
        if score is None:
            score = getattr(latest_frame, "levels_completed", 0)
        score = score or 0
        dscore = score - self.prev_score
        self.mem.model.observe(self.prev_state, self.prev_action, s2,
                               dscore, died)
        # reaction detection: our move made a NEW color appear near the
        # avatar (highlight/sparkle) — used only in desperation mode
        self._world_reacted = False
        if self.prev_action.x < 0 and self.prev_action.name != "RESET" \
                and self.prev_grid:
            cur_g0 = _last_layer(latest_frame)
            if cur_g0 and len(cur_g0) == len(self.prev_grid):
                prev_colors = {v for row in self.prev_grid for v in row}
                new_cells = [(x, y)
                             for y, (rp, rc) in enumerate(
                                 zip(self.prev_grid, cur_g0))
                             if rp != rc
                             for x, (a, b) in enumerate(zip(rp, rc))
                             if a != b and b not in prev_colors]
                if new_cells and len(new_cells) <= 40:
                    posr = self.avm.pos
                    if posr is None or any(
                            abs(cx - posr[0]) <= 4 and abs(cy - posr[1]) <= 4
                            for (cx, cy) in new_cells):
                        self._world_reacted = True
        # object-level learning (simple actions only)
        cur_grid = _last_layer(latest_frame)
        if self.prev_action.x < 0 and self.prev_action.name != "RESET":
            self.avm.update(self.prev_grid, cur_grid, self.prev_action.name,
                            ignore=self.mem.vol.mask)
        if dscore > 0:
            # colors that shrank when we scored are goal colors
            if self.prev_grid and cur_grid:
                pc, cc = _color_counts(self.prev_grid), _color_counts(cur_grid)
                for color, n in pc.items():
                    if cc.get(color, 0) < n:
                        self.goal_colors.add(color)
            # new level: per-level navigation state resets
            self.visited_cells.clear()
            self.exhausted_targets.clear()
            self.avm.blocked_tries.clear()
            self.avm.pos = None
            self.nav_target = None
            self._solver.clear()
            self._legend_order = []
            self._legend_box = None
            self._legend_at = -999
            # a plan/replay built for the previous level keeps executing
            # garbage into the new one until a mismatch is noticed
            self.plan.clear()
            self.plan_expected.clear()
            self.replay_queue.clear()
            self.momentum = None
            self.momentum_streak = 0
        # track novelty (+ momentum: keep repeating an action that finds
        # new states -- powerful for movement-style games)
        if s2 not in self.known_states:
            self.known_states.add(s2)
            self._novel_recent += 1
            self.since_new_state = 0
            new_m = self.prev_action if self.prev_action.x < 0 else None
            if new_m is None or self.momentum is None or \
                    new_m.name != self.momentum.name:
                self.momentum_streak = 0
            self.momentum = new_m
        else:
            self.since_new_state += 1
            self.momentum = None
            self.momentum_streak = 0
        # interface learning: does this action name ever do anything?
        nm = self.prev_action.name
        if nm != "RESET":
            if s2 != self.prev_state:
                self.mem.action_effect[nm] += 1
                self.mem.action_moves[nm] += 1
                # undo detection: did this exactly revert the previous
                # transition (a -> b then b -> a)?
                if self._last_transition is not None \
                        and self._last_transition == (s2, self.prev_state):
                    self.mem.action_revert[nm] += 1
                self._last_transition = (self.prev_state, s2)
            else:
                self.mem.action_noop[nm] += 1
        # remember productive clicks + click-color rule induction
        if self.prev_action.x >= 0:
            px, py = self.prev_action.x, self.prev_action.y
            if s2 != self.prev_state:
                self.mem.good_clicks.add((px, py))
            if self.prev_grid and 0 <= py < len(self.prev_grid) \
                    and 0 <= px < len(self.prev_grid[0]):
                c = self.prev_grid[py][px]
                if s2 != self.prev_state:
                    self.mem.click_color_good[c] += 1
                    self.mem.click_ctx_good[(px, py, c)] += 1
                else:
                    self.mem.click_color_bad[c] += 1
                    self.mem.click_ctx_bad[(px, py, c)] += 1
        if dscore > 0:
            self.last_scoreup_at = self.action_count
            self._telemetry(f"LEVELUP->{score}")
        # score-up: bank the episode prefix, keyed by the episode start state
        if dscore > 0 and self.episode_start_state is not None:
            prev_best = self.mem.best_prefix_by_start.get(self.episode_start_state)
            eff = self.episode_effects
            pruned = [a for i, a in enumerate(self.episode_actions)
                      if i >= len(eff) or eff[i]]
            if prev_best is None or score > prev_best[0]:
                # prune visible no-ops so post-death replays are short
                self.mem.best_prefix_by_start[self.episode_start_state] = \
                    (score, pruned)
            # CROSS-LEVEL TRANSFER (the lp85-cascade, industrialized):
            # the winning sequence of level N is the best prior for
            # level N+1 -- queue its tail for a one-shot attempt there,
            # and remember its last productive click coordinates as the
            # supreme click tier of the new level
            self._won_seq = pruned[-120:]
            self._won_clicks = [(a.x, a.y) for a in pruned
                                if a.x >= 0][-3:]
            # colors under the winning clicks: at the next level, click
            # objects of these colors WHEREVER they moved (absolute
            # coordinates miss relocated buttons)
            wc: set[int] = set()
            for (wx, wy) in self._won_clicks:
                for (cx, cy, c), n in self.mem.click_ctx_good.items():
                    if (cx, cy) == (wx, wy) and n > 0:
                        wc.add(c)
            self._won_click_colors = wc
            self._won_seq_tried = False
            # a score-up usually means a new level: rebind the episode start
            # to the upcoming state so future banking is per-level
            self.episode_start_state = None
            self.episode_actions = []
            self.episode_effects = []
        self.prev_score = score
        self.prev_action = None  # consumed

    def _start_episode(self) -> None:
        # dying mid strategy-replay means the strategy is bad here
        self._strategy_queue = deque()
        self.prev_state = None
        self.prev_action = None
        self.episode_actions = []
        self.episode_effects = []
        self.episode_start_state = None   # bound on the next observed frame
        self._last_transition = None
        self.plan.clear()
        self.plan_expected.clear()
        self.replay_queue = deque()       # loaded once the start state is known
        self.since_new_state = 0
        self.momentum = None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _bfs_plan(self, start: int,
                  available: list[GameAction]) -> Optional[list[ActionKey]]:
        """BFS over the learned state graph.

        Goal preference: state with a known score-up action, else nearest
        state with untried actions (frontier). Depth-limited for speed.

        Cost throttle: rebuilding the adjacency index is O(edges) PER
        CALL; with 10k+ learned states a single plan can take longer
        than the whole per-game time budget allows. On big graphs, plan
        only every 8th action (plans are followed multi-step anyway).
        """
        n_edges = len(self.mem.model.next)
        if n_edges > 4000 and self.action_count % 8 != 0 \
                and not self.mem.model.score_up:
            return None
        score_states = {ms for (ms, _a) in self.mem.model.score_up}
        parents: dict[int, tuple[int, ActionKey]] = {}
        seen = {start}
        q: deque[tuple[int, int]] = deque([(start, 0)])
        frontier_goal: Optional[int] = None

        # adjacency index once per plan (was O(nodes*edges) inside the loop)
        adj: dict[int, list[tuple[ActionKey, int]]] = defaultdict(list)
        for (ms, akey), outcomes in self.mem.model.next.items():
            if (ms, akey) in self.mem.model.deadly:
                continue
            nxt = max(outcomes, key=outcomes.get)  # type: ignore[arg-type]
            if nxt != ms:
                adj[ms].append((akey, nxt))

        while q:
            state, depth = q.popleft()
            if depth > 25:
                continue
            if state != start and state in score_states:
                return self._backtrack(parents, start, state)
            if frontier_goal is None and state != start \
                    and self._has_untried(state):
                frontier_goal = state
            for akey, nxt in adj.get(state, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    parents[nxt] = (state, akey)
                    q.append((nxt, depth + 1))

        if frontier_goal is not None:
            return self._backtrack(parents, start, frontier_goal)
        return None

    def _backtrack(self, parents: dict[int, tuple[int, ActionKey]],
                   start: int, goal: int) -> list[ActionKey]:
        path: list[ActionKey] = []
        cur = goal
        while cur != start:
            prev, akey = parents[cur]
            path.append(akey)
            cur = prev
        path.reverse()
        return path

    def _has_untried(self, state: int) -> bool:
        """Frontier test against actions this game actually offers."""
        tried = self.mem.tried.get(state, set())
        tried_simple = {t.name for t in tried if t.x < 0}
        if self.mem.simple_seen - tried_simple:
            return True
        if self.mem.complex_seen:
            n_complex = sum(1 for t in tried if t.x >= 0)
            return n_complex < MAX_COMPLEX_PER_STATE
        return False

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------
    def _pick_exploration_action(self, s: int, grid: list[list[int]],
                                 available: list[GameAction]
                                 ) -> Optional[ActionKey]:
        """Untried action at the current state, or None if exhausted here."""
        tried = self.mem.tried[s]
        model = self.mem.model

        # candidate simple actions first (cheap, often movement)
        simple = [a for a in available
                  if a is not GameAction.RESET and not a.is_complex()]
        candidates: list[ActionKey] = []
        for a in simple:
            akey = ActionKey(a)
            if akey in tried or self._suppressed(akey.name) \
                    or self._useless(akey.name):
                continue
            if model.is_known_noop(s, akey) or (s, akey) in model.deadly:
                continue
            candidates.append(akey)
        if candidates:
            fresh = [a for a in candidates if not self._reverty(a.name)]
            return self.rng.choice(fresh or candidates)

        # then ACTION6 with informed targets (bounded per state; if the game
        # offers ONLY clicks, allow a much deeper probe per state)
        complex_avail = [a for a in available if a.is_complex()]
        n_complex_tried = sum(1 for t in tried if t.x >= 0)
        click_cap = MAX_COMPLEX_PER_STATE if simple else 192
        if complex_avail and n_complex_tried < click_cap \
                and not self._useless("ACTION6"):
            good = self.mem.click_color_good
            bad = self.mem.click_color_bad

            def color_at(t: tuple[int, int]) -> Optional[int]:
                x, y = t
                if grid and 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                    return grid[y][x]
                return None

            def dead_color(t: tuple[int, int]) -> bool:
                c = color_at(t)
                return c is not None and good.get(c, 0) == 0 \
                    and bad.get(c, 0) >= 8

            # tiers: goal-colored objects > proven coords > productive-
            # color objects > symmetry breaks > structure. Goal colors
            # learned from earlier levels apply to CLICKING too.
            centroids = _object_centroids(grid)
            tierG = [t for t in centroids
                     if (color_at(t) or -1) in self.goal_colors]
            tier0 = list(self.mem.good_clicks)
            tierS = _symmetry_break_cells(grid)
            tier1 = [t for t in centroids
                     if good.get(color_at(t) or -1, 0) > 0 and t not in tierG]
            tier2 = _enclosed_cells(grid) + \
                [t for t in centroids if t not in tier1 and t not in tierG]
            if self.prev_grid and grid:
                tier2 += _diff_cells(self.prev_grid, grid)
            self.rng.shuffle(tierG)
            self.rng.shuffle(tier0)
            tier1.sort(key=lambda t: -(good.get(color_at(t) or -1, 0) + 1)
                       / (good.get(color_at(t) or -1, 0)
                          + bad.get(color_at(t) or -1, 0) + 2))
            self.rng.shuffle(tier2)
            tierT: list[tuple[int, int]] = []
            if self._desperate():
                tierT = _template_mismatch_cells(grid)
                self.rng.shuffle(tierT)
            tierW = [t for t in getattr(self, "_won_clicks", [])
                     if 0 <= t[0] < 64 and 0 <= t[1] < 64]
            wcc = getattr(self, "_won_click_colors", set())
            if wcc:
                tierW = tierW + [t for t in centroids
                                 if color_key(t) in wcc and t not in tierW]
            targets = tierW + tierG + tier0 + tierT[:24] + tier1 \
                + tierS[:16] + tier2
            if not targets:
                targets = [(x, y)
                           for x in range(0, 64, CLICK_GRID_STRIDE)
                           for y in range(0, 64, CLICK_GRID_STRIDE)]
                self.rng.shuffle(targets)
            ctx_g = self.mem.click_ctx_good
            ctx_b = self.mem.click_ctx_bad
            for (x, y) in targets:
                if dead_color((x, y)):
                    continue
                # contextual dead rule: this exact cell, showing this
                # exact color, has repeatedly done nothing
                if grid and 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                    ctx = (x, y, grid[y][x])
                    if ctx_g.get(ctx, 0) == 0 and ctx_b.get(ctx, 0) >= 4:
                        continue
                akey = ActionKey(complex_avail[0], x, y)
                if akey not in tried and not model.is_known_noop(s, akey) \
                        and (s, akey) not in model.deadly:
                    return akey
        return None

    def _fallback_action(self, s: int, available: list[GameAction]) -> ActionKey:
        """All local options exhausted and no plan: random safe legal action."""
        model = self.mem.model
        simple = [a for a in available
                  if a is not GameAction.RESET and not a.is_complex()]
        safe = [ActionKey(a) for a in simple
                if (s, ActionKey(a)) not in model.deadly
                and not model.is_known_noop(s, ActionKey(a))
                and not self._suppressed(a.name)
                and not self._useless(a.name)]
        if safe:
            fresh = [a for a in safe if not self._reverty(a.name)]
            pool = fresh or safe
            if self.last_scoreup_at:
                return self.rng.choice(pool)
            # never-scored game: prefer the action leading to the least-
            # visited known state (uniform random walked tr87's fully-
            # explored graph forever)
            def visits(ak: ActionKey) -> int:
                nxt = model.predicted_next(s, ak)
                return self.mem.visit_count.get(nxt, 0) \
                    if nxt is not None else -1
            lo = min(visits(ak) for ak in pool)
            return self.rng.choice([ak for ak in pool if visits(ak) == lo])
        non_deadly = [ActionKey(a) for a in simple
                      if (s, ActionKey(a)) not in model.deadly]
        if non_deadly:
            return self.rng.choice(non_deadly)
        complex_avail = [a for a in available if a.is_complex()]
        if complex_avail:
            # fallback clicks skip known-dead contexts instead of blind
            # uniform sampling
            grid = getattr(self, "_cur_grid", None)
            ctx_g = self.mem.click_ctx_good
            ctx_b = self.mem.click_ctx_bad
            for _ in range(10):
                x, y = self.rng.randrange(64), self.rng.randrange(64)
                if grid and 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                    ctx = (x, y, grid[y][x])
                    if ctx_g.get(ctx, 0) == 0 and ctx_b.get(ctx, 0) >= 4:
                        continue
                break
            return ActionKey(complex_avail[0], x, y)
        pool = [a for a in available if a is not GameAction.RESET] or \
            [GameAction.ACTION1]
        return ActionKey(self.rng.choice(pool))

    def _nav_action(self, s: int, grid: list[list[int]],
                    available: list[GameAction]) -> Optional[ActionKey]:
        dirmap = self.avm.direction_map()
        if len(dirmap) < 2 or not grid:
            return None
        pos = self.avm.locate(grid)
        if pos is None:
            return None
        self.visited_cells.add(pos)
        if self.nav_target is not None and pos == self.nav_target:
            # arrived: try every untried non-movement action here before
            # writing the spot off (interaction key varies per game)
            self.exhausted_targets.add(pos)
            self.nav_target = None
            move_names = set(dirmap)
            for a in available:
                if a is GameAction.RESET or a.is_complex() \
                        or a.name in move_names:
                    continue
                ak = ActionKey(a)
                if ak not in self.mem.tried[s] \
                        and (s, ak) not in self.mem.model.deadly:
                    return ak
            # desperation: try to OVERLAP the attractor itself — some
            # pickups trigger only on overlap, not adjacency (wa30)
            if self._desperate():
                attract = getattr(self, "_last_rare", set()) \
                    | self.goal_colors
                best_cell = None
                for dy2 in (-2, -1, 0, 1, 2):
                    for dx2 in (-2, -1, 0, 1, 2):
                        cx2, cy2 = pos[0] + dx2, pos[1] + dy2
                        if 0 <= cy2 < len(grid) \
                                and 0 <= cx2 < len(grid[0]) \
                                and grid[cy2][cx2] in attract:
                            best_cell = (cx2, cy2)
                            break
                    if best_cell:
                        break
                if best_cell is not None:
                    bd, ba = None, None
                    for name2, d2 in dirmap.items():
                        nd = abs(pos[0] + d2[0] - best_cell[0]) \
                            + abs(pos[1] + d2[1] - best_cell[1])
                        if bd is None or nd < bd:
                            bd, ba = nd, name2
                    if ba is not None:
                        ak2 = ActionKey(GameAction[ba])
                        if self._is_legal(ak2, available) \
                                and (s, ak2) not in self.mem.model.deadly:
                            return ak2
        target, path = self._pick_target_and_path(grid, pos, dirmap)
        if target is None or not path:
            return None
        self.nav_target = target
        name = path[0]
        akey = ActionKey(GameAction[name])
        if not self._is_legal(akey, available) \
                or (s, akey) in self.mem.model.deadly:
            return None
        return akey

    def _detect_legend(self, grid: list[list[int]]) -> None:
        """Sequence-legend reader (sk48-class): a horizontal row of >=3
        small colored squares in the top/bottom margin reads as VISIT
        THESE COLORS IN THIS ORDER. Sets _legend_order (colors, left to
        right) and _legend_box (region to exclude from targeting)."""
        self._legend_order: list[int] = []
        self._legend_box = None
        if not grid:
            return
        h, w = len(grid), len(grid[0])
        comps = _components_info(grid)
        av = self.avm.avatar_color()
        # candidates: small comps in the top or bottom quarter
        for band in ((0, h // 4), (3 * h // 4, h)):
            row_groups: dict[int, list[tuple[int, int, int, int]]] = {}
            for (cx, cy, n, c) in comps:
                if band[0] <= cy < band[1] and 2 <= n <= 40:
                    row_groups.setdefault(cy // 4, []).append((cx, cy, n, c))
            for _k, grp in row_groups.items():
                if len(grp) < 3:
                    continue
                grp.sort(key=lambda t: t[0])
                colors = [c for (_x, _y, _n, c) in grp
                          if c != av]
                # need >=3 distinct-ish colors in a horizontal line
                if len(colors) >= 3:
                    ys = [gy for (_x, gy, _n, _c) in grp]
                    if max(ys) - min(ys) <= 2:
                        self._legend_order = colors
                        xs = [gx for (gx, _y, _n, _c) in grp]
                        self._legend_box = (min(xs) - 3, min(ys) - 3,
                                            max(xs) + 3, max(ys) + 3)
                        return

    def _pick_target_and_path(self, grid: list[list[int]],
                              pos: tuple[int, int],
                              dirmap: dict[str, tuple[int, int]]
                              ) -> tuple[Optional[tuple[int, int]],
                                         list[str]]:
        """One BFS from pos; pick the best-category nearest cell."""
        h, w = len(grid), len(grid[0])
        if getattr(self, "_cur_counts", None) is None:
            self._cur_counts = _color_counts(grid)
        counts = self._cur_counts
        background = max(counts, key=counts.get)  # type: ignore[arg-type]
        avatar_color = self.avm.avatar_color()
        rare = {c for c, n in counts.items()
                if c != background and c != avatar_color and n <= 24}
        if self._desperate():
            # small-OBJECT colors too: multi-instance pickups (wa30's
            # three 12-cell candles total 36) never pass the total<=24 rule
            maxcomp: dict[int, int] = {}
            seen_sc = [[False] * w for _ in range(h)]
            for y0 in range(h):
                for x0 in range(w):
                    if seen_sc[y0][x0] or grid[y0][x0] == background:
                        continue
                    color0 = grid[y0][x0]
                    stack = [(x0, y0)]
                    seen_sc[y0][x0] = True
                    ncomp = 0
                    while stack:
                        cx0, cy0 = stack.pop()
                        ncomp += 1
                        for dx0, dy0 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx0, ny0 = cx0 + dx0, cy0 + dy0
                            if 0 <= nx0 < w and 0 <= ny0 < h \
                                    and not seen_sc[ny0][nx0] \
                                    and grid[ny0][nx0] == color0:
                                seen_sc[ny0][nx0] = True
                                stack.append((nx0, ny0))
                    if ncomp > maxcomp.get(color0, 0):
                        maxcomp[color0] = ncomp
            rare |= {c for c, n in counts.items()
                     if c != background and c != avatar_color
                     and n <= 90 and maxcomp.get(c, 99) <= 30}
        self._last_rare = rare
        blocked = {c for c, n in self.avm.blocked_tries.items() if n >= 2}
        wall_colors = self.avm.wall_colors()

        # a target is "reached" when the avatar's body would overlap it:
        # BFS cells within chebyshev 1 of a target-colored cell count
        # (2-cell steps put exact cells off-parity half the time)
        def near_set(colors: set[int]) -> set[tuple[int, int]]:
            out: set[tuple[int, int]] = set()
            if not colors:
                return out
            for y in range(h):
                for x in range(w):
                    if grid[y][x] in colors:
                        for dy2 in (-1, 0, 1):
                            for dx2 in (-1, 0, 1):
                                nx2, ny2 = x + dx2, y + dy2
                                if 0 <= nx2 < w and 0 <= ny2 < h:
                                    out.add((nx2, ny2))
            return out

        goal_set = {c for c in self.goal_colors if c != avatar_color}
        # legend-driven ordered visiting (desperation only): the current
        # legend color becomes THE goal; legend cells are instructions,
        # never targets
        legend_box = None
        if self._desperate():
            if self.action_count - getattr(self, "_legend_at", -999) > 300:
                self._legend_at = self.action_count
                self._detect_legend(grid)
            order = getattr(self, "_legend_order", [])
            legend_box = getattr(self, "_legend_box", None)
            if order and legend_box:
                lx0, ly0, lx1, ly1 = legend_box
                for c in order:
                    outside = any(
                        grid[y][x] == c
                        and not (lx0 <= x <= lx1 and ly0 <= y <= ly1)
                        for y in range(h) for x in range(w))
                    if outside:
                        goal_set = {c}
                        break
        goal_near = near_set(goal_set)
        rare_near = near_set(rare)
        if legend_box is not None:
            lx0, ly0, lx1, ly1 = legend_box
            legend_cells = {(x, y) for y in range(max(0, ly0), min(h, ly1 + 1))
                            for x in range(max(0, lx0), min(w, lx1 + 1))}
            goal_near -= legend_cells
            rare_near -= legend_cells

        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {pos}
        q: deque[tuple[int, int]] = deque([pos])
        goal_hit = rare_hit = frontier_hit = None
        while q:
            cur = q.popleft()
            x, y = cur
            if cur != pos and cur not in self.exhausted_targets:
                if goal_hit is None and cur in goal_near:
                    goal_hit = cur
                    break
                if rare_hit is None and cur in rare_near:
                    rare_hit = cur
                if frontier_hit is None and cur not in self.visited_cells:
                    frontier_hit = cur
            for name, (dx, dy) in dirmap.items():
                nxt = (x + dx, y + dy)
                if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                    continue
                if nxt in seen or nxt in blocked:
                    continue
                # every cell the step passes through must be passable
                n = max(abs(dx), abs(dy))
                ok = True
                for i in range(1, n + 1):
                    cx2 = x + round(dx * i / n)
                    cy2 = y + round(dy * i / n)
                    if grid[cy2][cx2] in wall_colors:
                        ok = False
                        break
                if not ok:
                    continue
                seen.add(nxt)
                prev[nxt] = (cur, name)
                q.append(nxt)

        target = goal_hit or rare_hit or frontier_hit
        if target is None:
            return None, []
        path: list[str] = []
        cur = target
        while cur != pos:
            p, name = prev[cur]
            path.append(name)
            cur = p
        path.reverse()
        return target, path

    def _record_available(self, available: list[GameAction]) -> None:
        for a in available:
            if a is GameAction.RESET:
                continue
            if a.is_complex():
                self.mem.complex_seen = True
            else:
                self.mem.simple_seen.add(a.name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_game_action(a: Any) -> Optional[GameAction]:
        """Normalize an available_actions entry (int, str, or GameAction).

        The real ARC-AGI-3-Agents framework sends available_actions as a
        list of ints; local mocks may send GameAction objects or names.
        """
        if isinstance(a, GameAction):
            return a
        # NOTE: GameAction values are (id, action_type) tuples internally,
        # so GameAction(6) raises ValueError. from_id() is the real API.
        # This silently failed in v1-v12: every frame fell back to "all 7
        # actions available" and most of the budget probed phantom actions.
        try:
            return GameAction.from_id(int(a))
        except (ValueError, TypeError):
            pass
        try:
            return GameAction[str(a).upper()]
        except KeyError:
            return None

    def _available_actions(self, frame: FrameData) -> list[GameAction]:
        avail = getattr(frame, "available_actions", None)
        out: list[GameAction] = []
        for a in (avail or []):
            ga = self._to_game_action(a)
            if ga is not None:
                out.append(ga)
        if out:
            return out
        return [a for a in GameAction if a is not GameAction.RESET]

    def _can_reset(self, frame: FrameData) -> bool:
        avail = getattr(frame, "available_actions", None)
        if not avail:
            return True
        return GameAction.RESET in self._available_actions(frame)

    USELESS_AFTER_SIMPLE = 15
    USELESS_AFTER_CLICK = 60

    def _reverty(self, name: str) -> bool:
        """Action mostly undoes the previous transition (an undo button)."""
        moves = self.mem.action_moves.get(name, 0)
        return moves >= 10 and \
            self.mem.action_revert.get(name, 0) / moves > 0.5

    def _useless(self, name: str) -> bool:
        """Action name is a no-op in (almost) every state it was tried.

        RATIO-based, not ever-had-an-effect: per-action counter pixels
        tick BEFORE the volatility mask forms, so every action registered
        one early "effect" and the old filter never fired (lf52 pressed a
        do-nothing ACTION7 969 times through fallback). Exploration and
        fallback skip useless actions; exploit, replay, and nav-arrival
        paths deliberately do NOT (an action can matter only in rare
        states, e.g. 'use key at door')."""
        eff = self.mem.action_effect.get(name, 0)
        noop = self.mem.action_noop.get(name, 0)
        thr = self.USELESS_AFTER_CLICK if name == "ACTION6" \
            else self.USELESS_AFTER_SIMPLE
        if eff + noop < thr:
            return False
        return noop / (eff + noop) >= 0.95

    def _suppressed(self, name: str) -> bool:
        return self.suppress_until.get(name, 0) > self.action_count

    def _break_fixation(self) -> None:
        """If one action dominates the recent window with no score-up, ban
        it for a while. Kills 'press ACTION5 4000 times' pathologies where an
        on-screen counter makes every press look like a novel state."""
        if len(self.recent_actions) < FIXATION_WINDOW:
            return
        if self.action_count - self.last_scoreup_at < FIXATION_WINDOW:
            return
        counts: dict[str, int] = {}
        for n in self.recent_actions:
            counts[n] = counts.get(n, 0) + 1
        name, cnt = max(counts.items(), key=lambda kv: kv[1])
        if cnt >= FIXATION_SHARE * FIXATION_WINDOW:
            self.suppress_until[name] = self.action_count + SUPPRESS_FOR
            self.recent_actions.clear()
            if self.momentum is not None and self.momentum.name == name:
                self.momentum = None
                self.momentum_streak = 0

    def _is_legal(self, akey: ActionKey,
                  available: list[GameAction]) -> bool:
        return any(a.name == akey.name for a in available)

    # ------------------------------------------------------------------
    # Telemetry: compact lines in the competition-rerun log so scored
    # submissions double as diagnostics of the PRIVATE eval games (the
    # only feedback channel — episode replays are not exposed).
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SOLVER MODE — collect archetype (v69)
    # Detection: avatar with a known direction map + >=2 small solid
    # same-colored targets. Execution: nearest target, greedy stepping
    # with perpendicular sidesteps on blocks (the manual-play pattern
    # that collected wa30 candles at baseline pace), interact keys on
    # any reaction/overlap, advance when the target count drops.
    # Bounded: activates on stagnation, aborts after fruitless work.
    # ------------------------------------------------------------------
    SOLVER_STAGNANT = 600     # actions without score-up before activating
    SOLVER_PATIENCE = 500     # fruitless solver actions before abort

    def _solver_detect(self, grid: list[list[int]]) -> Optional[dict]:
        dirmap = self.avm.direction_map()
        if len(dirmap) < 3 or not grid:
            return None
        av_color = self.avm.avatar_color()
        h, w = len(grid), len(grid[0])
        counts = _color_counts(grid)
        background = max(counts, key=counts.get)  # type: ignore[arg-type]
        # single-color groups of small solid components
        groups: dict[int, list[tuple[int, int]]] = {}
        seen: set[tuple[int, int]] = set()
        for y in range(h):
            for x in range(w):
                if (x, y) in seen or grid[y][x] == background:
                    continue
                c0 = grid[y][x]
                stack = [(x, y)]
                seen.add((x, y))
                cells = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h \
                                and (nx, ny) not in seen \
                                and grid[ny][nx] == c0:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                if c0 == av_color or not (4 <= len(cells) <= 40):
                    continue
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                if len(cells) < 0.7 * (max(xs) - min(xs) + 1) \
                        * (max(ys) - min(ys) + 1):
                    continue
                groups.setdefault(c0, []).append(
                    (sum(xs) // len(cells), sum(ys) // len(cells)))
        for c0, cents in groups.items():
            if 2 <= len(cents) <= 8:
                return {"color": c0, "targets": cents}
        return None

    def _solver_action(self, s: int, grid: list[list[int]],
                       available: list[GameAction]) -> Optional[ActionKey]:
        st = self._solver
        stagnant = self.action_count - self.last_scoreup_at
        if not st.get("active"):
            if stagnant < self.SOLVER_STAGNANT \
                    or st.get("fails", {}).get(self.prev_score, 0) >= 2:
                return None
            det = self._solver_detect(grid)
            if det is None:
                return None
            # EVIDENCE GATE: the world must have confirmed the archetype
            # before the solver commits budget — either a target of this
            # color already vanished during exploration, or a reaction
            # fired while the avatar was near one. Ungated activation
            # displaced productive exploration and cost 5 levels (v69).
            base = st.get("baseline_counts", {})
            c0 = det["color"]
            if c0 not in base:
                base[c0] = len(det["targets"])
                st["baseline_counts"] = base
            confirmed = len(det["targets"]) < base.get(c0, 99) \
                or st.get("react_near", False)
            if self._world_reacted:
                pos0 = self.avm.pos
                if pos0 is not None and any(
                        abs(tx0 - pos0[0]) + abs(ty0 - pos0[1]) <= 6
                        for (tx0, ty0) in det["targets"]):
                    st["react_near"] = True
                    confirmed = True
            if not confirmed:
                return None
            st.update(active=True, color=det["color"], spent=0,
                      count=len(det["targets"]), sidestep=0)
        # re-detect targets every 4th frame (full-grid scan is costly)
        if st["spent"] % 4 == 0 or st.get("cached_det") is None:
            st["cached_det"] = self._solver_detect(grid)
        det = st["cached_det"]
        st["spent"] += 1
        if st["spent"] > self.SOLVER_PATIENCE:
            fails = st.get("fails", {})
            fails[self.prev_score] = fails.get(self.prev_score, 0) + 1
            st.clear()
            st["fails"] = fails                 # bounded retries per level
            return None
        if det is None or det["color"] != st.get("color"):
            # all targets gone (or scene changed): success or moot
            st.clear()
            return None
        if len(det["targets"]) < st.get("count", 99):
            st["count"] = len(det["targets"])
            st["spent"] = 0                     # progress: reset patience
        dirmap = self.avm.direction_map()
        pos = self.avm.locate(grid)
        if pos is None or not dirmap:
            st.clear()
            return None
        # a reaction near us (highlight): press interact keys first
        if self._world_reacted:
            move_names = set(dirmap)
            for a in available:
                if a is GameAction.RESET or a.is_complex() \
                        or a.name in move_names:
                    continue
                ak = ActionKey(a)
                if ak not in self.mem.tried[s] \
                        and (s, ak) not in self.mem.model.deadly:
                    return ak
        # greedy step toward the nearest target with sidestep-on-block
        # and cycling APPROACH MODES: targets are often solid from some
        # sides (wa30 candles admit entry only from below)
        tx, ty = min(det["targets"],
                     key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1]))
        if self._last_solver_pos == pos:
            st["sidestep"] += 1
        else:
            st["sidestep"] = max(0, st.get("sidestep", 0) - 1)
        self._last_solver_pos = pos
        if st.get("sidestep", 0) >= 8:
            st["mode"] = (st.get("mode", 0) + 1) % 4
            st["sidestep"] = 0
            st["spent"] = max(0, st["spent"] - 100)  # new approach, new hope
        mode = st.get("mode", 0)
        if mode == 1:
            # FROM-BELOW routine (three phases, the manual wa30 route):
            # descend to open air, align the column, rise into the target
            if abs(pos[0] - tx) > 1 and pos[1] < ty + 6:
                tx, ty = pos[0], min(63, ty + 8)      # descend
            elif abs(pos[0] - tx) > 1:
                tx, ty = tx, pos[1]                    # align column
            # else: rise straight into the target
        elif mode in (2, 3):
            off = ((-6, 0), (6, 0))[mode - 2]
            wx = max(0, min(63, tx + off[0]))
            wy = max(0, min(63, ty + off[1]))
            if abs(pos[0] - wx) + abs(pos[1] - wy) > 3:
                tx, ty = wx, wy
        dxx, dyy = tx - pos[0], ty - pos[1]
        prefer_x = abs(dxx) >= abs(dyy)
        if st.get("sidestep", 0) % 3 == 2:
            prefer_x = not prefer_x            # blocked: try the other axis
        want = (1 if dxx > 0 else -1, 0) if prefer_x \
            else (0, 1 if dyy > 0 else -1)
        best_name, best_dot = None, -99
        for name, d in dirmap.items():
            sx = 1 if d[0] > 0 else (-1 if d[0] < 0 else 0)
            sy = 1 if d[1] > 0 else (-1 if d[1] < 0 else 0)
            dot = sx * want[0] + sy * want[1]
            if dot > best_dot:
                best_dot, best_name = dot, name
        if best_name is None:
            st.clear()
            return None
        ak = ActionKey(GameAction[best_name])
        if self._is_legal(ak, available) \
                and (s, ak) not in self.mem.model.deadly:
            return ak
        return None

    def _desperate(self) -> bool:
        """Zero score after DESPERATE_AFTER actions: nothing to lose.

        The extended arsenal (react-interact, small-object targeting,
        overlap arrival, copy-task clicks) runs ONLY here — every one of
        these mechanisms helped some stuck game but cost levels when
        allowed to disturb already-working trajectories (v55-v57)."""
        return self.prev_score == 0 and self.action_count > DESPERATE_AFTER

    def _rc(self, why: str) -> None:
        key = why.split(":")[0]
        self._reason_counts[key] = self._reason_counts.get(key, 0) + 1

    def _telemetry(self, tag: str) -> None:
        try:
            import time as _t
            if not hasattr(self, "_t0"):
                self._t0 = _t.time()
            el = int(_t.time() - self._t0)
            rc = self._reason_counts
            mix = ",".join(f"{k[:2]}{v}" for k, v in
                           sorted(rc.items(), key=lambda kv: -kv[1])[:6])
            dm = self.avm.direction_map()
            print(f"[MYA]{tag} g={self.game_id} t={el}s a={self.action_count} "
                  f"lv={self.prev_score} st={len(self.known_states)} "
                  f"av={self.avm.avatar_color()} dm={len(dm)} "
                  f"ph={self.phase} clicks+{sum(self.mem.click_ctx_good.values())}"
                  f"-{sum(self.mem.click_ctx_bad.values())} mix={mix}",
                  flush=True)
        except Exception:
            pass

    def _emit(self, akey: ActionKey, s: int, why: str) -> GameAction:
        self._rc(why)
        action = GameAction[akey.name]
        if action.is_complex():
            x = akey.x if akey.x >= 0 else self.rng.randrange(64)
            y = akey.y if akey.y >= 0 else self.rng.randrange(64)
            akey = ActionKey(action, x, y)
            action.set_data({"x": x, "y": y})
            action.reasoning = {"why": why, "x": x, "y": y}
        else:
            action.reasoning = f"{why}"
        # bookkeeping for learning on the next frame
        self.recent_actions.append(akey.name)
        self.mem.tried[s].add(akey)
        self.mem.visit_count[s] += 1
        self.prev_state = s
        self.prev_action = akey
        self.prev_grid = getattr(self, "_cur_grid", self.prev_grid)
        self.episode_actions.append(akey)
        return action
