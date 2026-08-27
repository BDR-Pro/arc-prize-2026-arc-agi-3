"""LLM rescue-agent for ARC-AGI-3 (v2): v79 drives, LLM rescues stalls.

Inverted control vs v1:
  - The proven programmatic agent (v79) DRIVES every frame. It is fast and
    good at the games it handles.
  - The LLM is invoked ONLY when v79 has stalled (no score-up for
    STALL_TRIGGER actions). Expensive reasoning is spent exactly where the
    programmatic agent fails.
  - When invoked, the LLM may either name actions directly OR write Python
    code in a sandbox that inspects the board and calls act(...) — letting
    it run BFS / search, the mechanism that separates strong LLM agents.
  - Any error, empty result, or unproductive stint returns control to v79.
    v79 is a hard floor: the LLM can only add value.
"""
from __future__ import annotations
import re
import sys
import time
_MODULE_START = time.time()
LLM_GLOBAL_DEADLINE_S = 21600  # 6h: after this, pure v79 (rerun-time safety)
from collections import deque
from pathlib import Path
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import render as _render          # noqa: E402
import llm_client as _llmc        # noqa: E402

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "prog_agent", str(_HERE.parent / "my_agent.py"))
_prog = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_prog)
ProgAgent = _prog.MyAgent


SYSTEM_PROMPT = (
    "You are an expert agent for a 64x64 grid puzzle game. A fast rule-based "
    "player is STUCK on the current level and has handed control to you to "
    "find a breakthrough. You see the board as objects plus an ASCII color "
    "grid, the valid actions, recent history, and what the last action "
    "changed.\n"
    "Key guidance (hard-won):\n"
    "- A long thin strip of blocks flush against an edge is usually a "
    "timer/HUD bar, NOT clickable pieces. Do not click through it segment by "
    "segment.\n"
    "- Entities are connected same-color shapes. Actions' meanings vary per "
    "game; infer them from what changed.\n"
    "- When the goal is clear but the action order is not, SEARCH: write a "
    "BFS/greedy plan in code.\n"
    "You have two ways to act. Prefer CODE when search or computation helps:\n"
    "1) Direct: reply\n"
    "PLAN: <one line>\n"
    "ACTIONS: <comma list, e.g. ACTION1, ACTION1, ACTION6 12 30>\n"
    "   (ACTION6 needs 'ACTION6 <row> <col>').\n"
    "2) Code: reply with a fenced python block. Inside it you have `grid` "
    "(list[list[int]] rows), `objects` (list of dicts with color,n,center,"
    "bbox,shape), `valid` (list of action-name strings), and you MUST build "
    "a list `plan` of actions, each either 'ACTIONX' or ('ACTION6', row, "
    "col). Example:\n"
    "```python\n"
    "plan = []\n"
    "for o in objects:\n"
    "    if o['n'] <= 4:\n"
    "        plan.append(('ACTION6', o['center'][0], o['center'][1]))\n"
    "plan = plan[:6]\n"
    "```\n"
    "Keep plans <= 8 actions. Output ONLY one of the two formats."
)

_ACT_RE = re.compile(r"ACTION\s*([1-7])(?:\s+(\d+)\s+(\d+))?", re.I)
_RESET_RE = re.compile(r"\bRESET\b", re.I)
_CODE_RE = re.compile(r"```(?:python)?\s*(.+?)```", re.S | re.I)

_SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "abs": abs,
    "sorted": sorted, "sum": sum, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "any": any, "all": all, "int": int, "float": float,
    "bool": bool, "str": str, "round": round, "reversed": reversed,
}


class LLMAgent(Agent):
    MAX_ACTIONS = 16000
    STALL_TRIGGER = 250     # prog actions w/o score-up -> call the LLM
    LLM_SEQ_MAX = 8
    RESCUE_BUDGET = 120     # actions the LLM stint gets before back to prog
    LLM_FAIL_CAP = 6        # total LLM failures -> stop trying the LLM
    MAX_LLM_CALLS = 12      # per-game cap (shared model, 110 concurrent games)
    REQUERY_GAP = 6         # min actions between LLM queries within a rescue

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.prog = ProgAgent(*a, **k)
        try:
            self.client = _llmc.make_client()
        except Exception:                        # noqa: BLE001
            self.client = None
        self.queue: deque = deque()
        self.history: list[str] = []
        self.prev_grid: list[list[int]] = []
        self.prev_score = 0
        self.last_scoreup = 0
        self.n = 0
        self.llm_fails = 0
        self.rescue_start = -10 ** 9
        self.rescue_score0 = 0
        self.n_llm_calls = 0
        self._last_query = -10 ** 9
        self.n_llm_actions = 0

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        self.n += 1
        try:
            prog_choice = self.prog.choose_action(frames, latest_frame)
        except Exception:                        # noqa: BLE001
            prog_choice = None

        grid = self._grid(latest_frame)
        score = getattr(latest_frame, "levels_completed", 0) or 0
        if score > self.prev_score:
            self.last_scoreup = self.n
            self.queue.clear()                   # progressed: drop LLM plan
        self.prev_score = score

        in_rescue = self.n - self.rescue_start < self.RESCUE_BUDGET
        # end a rescue stint early if it produced a score-up
        if in_rescue and score > self.rescue_score0:
            in_rescue = False
            self.queue.clear()

        want_llm = (
            self.client is not None
            and time.time() - _MODULE_START < LLM_GLOBAL_DEADLINE_S
            and self.llm_fails < self.LLM_FAIL_CAP
            and self.n_llm_calls < self.MAX_LLM_CALLS
            and (in_rescue or (self.n - self.last_scoreup) >= self.STALL_TRIGGER)
        )

        if want_llm:
            if not in_rescue and not self.queue:
                # start a new rescue stint
                self.rescue_start = self.n
                self.rescue_score0 = score
                self._query_llm(grid, self._avail(latest_frame))
            if not self.queue and in_rescue                     and self.n - self._last_query >= self.REQUERY_GAP:
                self._query_llm(grid, self._avail(latest_frame))
            avail = self._avail(latest_frame)
            while self.queue:
                act = self.queue.popleft()
                if any(x.name == act.name for x in avail):
                    self.history.append(act.name)
                    self.prev_grid = grid
                    self.n_llm_actions += 1
                    return act
            # nothing usable this frame; fall through to prog

        self.prev_grid = grid
        return prog_choice if prog_choice is not None \
            else self._safe_default(latest_frame)

    # ---- LLM query ---------------------------------------------------
    def _query_llm(self, grid, avail) -> None:
        try:
            self.n_llm_calls += 1
            self._last_query = self.n
            last_change = None
            if self.prev_grid and grid and len(self.prev_grid) == len(grid):
                last_change = sum(
                    1 for r0, r1 in zip(self.prev_grid, grid)
                    for a, b in zip(r0, r1) if a != b)
            names = [a.name + ("(row col)" if a.is_complex() else "")
                     for a in avail]
            obs = _render.render_observation(grid, names, self.history,
                                             last_change)
            reply = self.client.chat(SYSTEM_PROMPT, obs, max_tokens=384)
            q = self._from_code(reply, grid, avail)
            if not q:
                q = self._parse(reply, avail)
            self.queue = q
            self.llm_fails = 0 if q else self.llm_fails + 1
        except Exception:                        # noqa: BLE001
            self.llm_fails += 1
            self.queue = deque()

    def _from_code(self, reply, grid, avail) -> deque:
        m = _CODE_RE.search(reply)
        if not m:
            return deque()
        code = m.group(1)
        objs = _render.segment(grid)
        # translate letter colors back to ints for convenience
        env = {"grid": grid, "objects": objs,
               "valid": [a.name for a in avail], "plan": []}
        try:
            exec(compile(code, "<llm>", "exec"),  # noqa: S102
                 {"__builtins__": _SAFE_BUILTINS}, env)
        except Exception:                         # noqa: BLE001
            return deque()
        return self._plan_to_actions(env.get("plan", []), avail)

    def _plan_to_actions(self, plan, avail) -> deque:
        legal = {a.name for a in avail}
        out: deque = deque()
        if not isinstance(plan, (list, tuple)):
            return out
        for item in plan:
            name = None
            row = col = None
            if isinstance(item, str):
                name = item.strip().upper()
            elif isinstance(item, (list, tuple)) and item:
                name = str(item[0]).strip().upper()
                if len(item) >= 3:
                    try:
                        row, col = int(item[1]), int(item[2])
                    except Exception:            # noqa: BLE001
                        row = col = None
            if not name or name not in legal:
                continue
            act = GameAction[name]
            if act.is_complex():
                if row is None:
                    continue
                row = max(0, min(63, row)); col = max(0, min(63, col))
                act.set_data({"x": col, "y": row})
            out.append(act)
            if len(out) >= self.LLM_SEQ_MAX:
                break
        return out

    def _parse(self, reply, avail) -> deque:
        legal = {a.name for a in avail}
        line = reply
        m = re.search(r"ACTIONS:\s*(.+)", reply, re.I | re.S)
        if m:
            line = m.group(1)
        out: deque = deque()
        for tok in line.split(","):
            tok = tok.strip()
            if _RESET_RE.search(tok):
                if "RESET" in legal:
                    out.append(GameAction.RESET)
                continue
            am = _ACT_RE.search(tok)
            if not am:
                continue
            name = f"ACTION{am.group(1)}"
            if name not in legal:
                continue
            act = GameAction[name]
            if act.is_complex():
                if am.group(2) is not None:
                    row, col = int(am.group(2)), int(am.group(3))
                    row = max(0, min(63, row)); col = max(0, min(63, col))
                    act.set_data({"x": col, "y": row})
                else:
                    continue
            out.append(act)
            if len(out) >= self.LLM_SEQ_MAX:
                break
        return out

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _grid(frame) -> list:
        g = getattr(frame, "frame", None) or []
        return [list(r) for r in g[-1]] if g else []

    def _avail(self, frame) -> list:
        raw = getattr(frame, "available_actions", None) or []
        out = []
        for a in raw:
            try:
                out.append(GameAction.from_id(int(a)))
            except Exception:                    # noqa: BLE001
                pass
        return out or [a for a in GameAction if a is not GameAction.RESET]

    def _safe_default(self, frame) -> GameAction:
        for a in self._avail(frame):
            if a is not GameAction.RESET:
                return a
        return GameAction.RESET


MyAgent = LLMAgent
