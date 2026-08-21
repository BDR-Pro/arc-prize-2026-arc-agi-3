"""ARC-AGI-3 agent (v10): world model + programmatic memory + planning + efficiency.

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
MAX_ACTIONS_PER_GAME = 4000     # generous budget; completing levels beats efficiency
NOOP_SKIP_THRESHOLD = 2        # times an action must no-op before we skip it
STUCK_WINDOW = 50              # actions with no new state before we force reset
MAX_COMPLEX_PER_STATE = 24
MOMENTUM_CAP = 60              # max consecutive repeats of one action
FIXATION_WINDOW = 200          # look-back for action-fixation detection
FIXATION_SHARE = 0.85          # one action dominating window w/o score-up
SUPPRESS_FOR = 120             # actions to ban a fixated action
RESEED_AFTER = 1000            # actions w/o score-up -> new exploration phase
CLICK_GRID_STRIDE = 8          # fallback coarse grid for ACTION6 exploration


def _grid_hash(frame: FrameData) -> int:
    """Stable hash of the observable game state.

    Only the LAST grid layer is hashed: games emit intermediate animation
    layers, and hashing them fragments identical logical states into
    thousands of unique hashes, destroying the world model's state graph.
    """
    grids = getattr(frame, "frame", None) or []
    if not grids:
        return 0
    return hash(tuple(tuple(row) for row in grids[-1]))


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
        self.blocked_tries: dict[tuple[int, int], int] = defaultdict(int)

    def update(self, prev_grid: list[list[int]], cur_grid: list[list[int]],
               action_name: str) -> None:
        if not prev_grid or not cur_grid or len(prev_grid) != len(cur_grid):
            return
        prev_objs = self._objects(prev_grid)
        cur_objs = self._objects(cur_grid)
        moved_any = False
        for key, ppos in prev_objs.items():
            cpos = cur_objs.get(key)
            if cpos is None:
                continue
            dx, dy = cpos[0] - ppos[0], cpos[1] - ppos[1]
            if dx == 0 and dy == 0:
                continue
            if max(abs(dx), abs(dy)) > 3:
                continue
            color = key[0]
            self.stats[(color, action_name)][(dx, dy)] += 1
            moved_any = True
        if not moved_any and self.pos is not None:
            dirs = self.direction_map()
            d = dirs.get(action_name)
            if d is not None:
                cell = (self.pos[0] + d[0], self.pos[1] + d[1])
                self.blocked_tries[cell] += 1

    def _objects(self, grid: list[list[int]]
                 ) -> dict[tuple[int, int], tuple[int, int]]:
        """(color, size) -> centroid, only for colors forming ONE component."""
        counts = _color_counts(grid)
        background = max(counts, key=counts.get)  # type: ignore[arg-type]
        h, w = len(grid), len(grid[0])
        comps: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        seen = [[False] * w for _ in range(h)]
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
                comps[color].append((xs // n, ys // n, n))
        out: dict[tuple[int, int], tuple[int, int]] = {}
        for color, lst in comps.items():
            if len(lst) == 1:
                cx, cy, n = lst[0]
                out[(color, n)] = (cx, cy)
        return out

    def avatar_color(self) -> Optional[int]:
        best_color, best_score = None, 0
        by_color: dict[int, int] = defaultdict(int)
        for (color, _act), deltas in self.stats.items():
            total = sum(deltas.values())
            top = max(deltas.values())
            if total >= self.MIN_OBS and top / total >= self.CONSISTENCY:
                by_color[color] += top
        for color, score in by_color.items():
            if score > best_score:
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
            if total >= self.MIN_OBS and top / total >= self.CONSISTENCY \
                    and max(abs(delta[0]), abs(delta[1])) == 1:
                out[act] = delta
        return out

    def locate(self, grid: list[list[int]]) -> Optional[tuple[int, int]]:
        color = self.avatar_color()
        if color is None or not grid:
            return None
        cells = [(x, y) for y, row in enumerate(grid)
                 for x, v in enumerate(row) if v == color]
        if not cells:
            return None
        if self.pos is not None:
            cells.sort(key=lambda c: abs(c[0] - self.pos[0]) +
                       abs(c[1] - self.pos[1]))
        cx = sum(c[0] for c in cells[:4]) // min(len(cells), 4)
        cy = sum(c[1] for c in cells[:4]) // min(len(cells), 4)
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

        # -- Reset handling ------------------------------------------------
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._learn(latest_frame, died=latest_frame.state is GameState.GAME_OVER)
            self._start_episode()
            action = GameAction.RESET
            action.reasoning = "reset: start/retry episode"
            return action

        # -- Learn from the outcome of the previous action -----------------
        self._learn(latest_frame, died=False)

        s = _grid_hash(latest_frame)
        grid = _last_layer(latest_frame)
        self._cur_grid = grid
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
    def _learn(self, latest_frame: FrameData, died: bool) -> None:
        if self.prev_state is None or self.prev_action is None:
            return
        s2 = _grid_hash(latest_frame)
        score = getattr(latest_frame, "score", None)
        if score is None:
            score = getattr(latest_frame, "levels_completed", 0)
        score = score or 0
        dscore = score - self.prev_score
        self.mem.model.observe(self.prev_state, self.prev_action, s2,
                               dscore, died)
        # object-level learning (simple actions only)
        cur_grid = _last_layer(latest_frame)
        if self.prev_action.x < 0 and self.prev_action.name != "RESET":
            self.avm.update(self.prev_grid, cur_grid, self.prev_action.name)
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
        # track novelty (+ momentum: keep repeating an action that finds
        # new states -- powerful for movement-style games)
        if s2 not in self.known_states:
            self.known_states.add(s2)
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
        # remember productive clicks
        if self.prev_action.x >= 0 and s2 != self.prev_state:
            self.mem.good_clicks.add((self.prev_action.x, self.prev_action.y))
        if dscore > 0:
            self.last_scoreup_at = self.action_count
        # score-up: bank the episode prefix, keyed by the episode start state
        if dscore > 0 and self.episode_start_state is not None:
            prev_best = self.mem.best_prefix_by_start.get(self.episode_start_state)
            if prev_best is None or score > prev_best[0]:
                self.mem.best_prefix_by_start[self.episode_start_state] = \
                    (score, list(self.episode_actions))
            # a score-up usually means a new level: rebind the episode start
            # to the upcoming state so future banking is per-level
            self.episode_start_state = None
            self.episode_actions = []
        self.prev_score = score
        self.prev_action = None  # consumed

    def _start_episode(self) -> None:
        self.prev_state = None
        self.prev_action = None
        self.episode_actions = []
        self.episode_start_state = None   # bound on the next observed frame
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
        """
        score_states = {ms for (ms, _a) in self.mem.model.score_up}
        parents: dict[int, tuple[int, ActionKey]] = {}
        seen = {start}
        q: deque[tuple[int, int]] = deque([(start, 0)])
        frontier_goal: Optional[int] = None

        while q:
            state, depth = q.popleft()
            if depth > 25:
                continue
            if state != start and state in score_states:
                return self._backtrack(parents, start, state)
            if frontier_goal is None and state != start \
                    and self._has_untried(state):
                frontier_goal = state
            for (ms, akey), outcomes in self.mem.model.next.items():
                if ms != state or (ms, akey) in self.mem.model.deadly:
                    continue
                nxt = max(outcomes, key=outcomes.get)  # type: ignore[arg-type]
                if nxt not in seen and nxt != state:
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
            if akey in tried or self._suppressed(akey.name):
                continue
            if model.is_known_noop(s, akey) or (s, akey) in model.deadly:
                continue
            candidates.append(akey)
        if candidates:
            return self.rng.choice(candidates)

        # then ACTION6 with informed targets (bounded per state; if the game
        # offers ONLY clicks, allow a much deeper probe per state)
        complex_avail = [a for a in available if a.is_complex()]
        n_complex_tried = sum(1 for t in tried if t.x >= 0)
        click_cap = MAX_COMPLEX_PER_STATE if simple else 96
        if complex_avail and n_complex_tried < click_cap:
            targets: list[tuple[int, int]] = []
            targets.extend(self.mem.good_clicks)
            targets.extend(_enclosed_cells(grid))
            targets.extend(_object_centroids(grid))
            if self.prev_grid and grid:
                targets.extend(_diff_cells(self.prev_grid, grid))
            if not targets:
                targets = [(x, y)
                           for x in range(0, 64, CLICK_GRID_STRIDE)
                           for y in range(0, 64, CLICK_GRID_STRIDE)]
            self.rng.shuffle(targets)
            for (x, y) in targets:
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
                and not self._suppressed(a.name)]
        if safe:
            return self.rng.choice(safe)
        non_deadly = [ActionKey(a) for a in simple
                      if (s, ActionKey(a)) not in model.deadly]
        if non_deadly:
            return self.rng.choice(non_deadly)
        complex_avail = [a for a in available if a.is_complex()]
        if complex_avail:
            return ActionKey(complex_avail[0],
                             self.rng.randrange(64), self.rng.randrange(64))
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
            # arrived: try interacting once before writing the spot off
            self.exhausted_targets.add(pos)
            self.nav_target = None
            for a in available:
                if a.name == "ACTION5":
                    akey5 = ActionKey(a)
                    if akey5 not in self.mem.tried[s] \
                            and (s, akey5) not in self.mem.model.deadly:
                        return akey5
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

    def _pick_target_and_path(self, grid: list[list[int]],
                              pos: tuple[int, int],
                              dirmap: dict[str, tuple[int, int]]
                              ) -> tuple[Optional[tuple[int, int]],
                                         list[str]]:
        """One BFS from pos; pick the best-category nearest cell."""
        h, w = len(grid), len(grid[0])
        counts = _color_counts(grid)
        background = max(counts, key=counts.get)  # type: ignore[arg-type]
        avatar_color = self.avm.avatar_color()
        rare = {c for c, n in counts.items()
                if c != background and c != avatar_color and n <= 24}
        blocked = {c for c, n in self.avm.blocked_tries.items() if n >= 2}

        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {pos}
        q: deque[tuple[int, int]] = deque([pos])
        goal_hit = rare_hit = frontier_hit = None
        while q:
            cur = q.popleft()
            x, y = cur
            if cur != pos and cur not in self.exhausted_targets:
                color = grid[y][x]
                if goal_hit is None and color in self.goal_colors \
                        and color != avatar_color:
                    goal_hit = cur
                    break
                if rare_hit is None and color in rare:
                    rare_hit = cur
                if frontier_hit is None and cur not in self.visited_cells:
                    frontier_hit = cur
            for name, (dx, dy) in dirmap.items():
                nxt = (x + dx, y + dy)
                if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                    continue
                if nxt in seen or nxt in blocked:
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
        try:
            return GameAction(a)          # int id
        except (ValueError, KeyError):
            pass
        try:
            return GameAction[str(a)]     # name string
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

    def _emit(self, akey: ActionKey, s: int, why: str) -> GameAction:
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
