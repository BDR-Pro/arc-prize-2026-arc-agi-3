"""LLM tool-agent for ARC-AGI-3 with a warm programmatic fallback.

Design:
  - The LLM proposes a SHORT action sequence per query (inference is slow;
    we cannot call it every action of a 16k budget).
  - The sequence is executed action-by-action.
  - A programmatic agent (v79) is fed every frame so its world model stays
    warm; it takes over whenever the LLM is unavailable, errors, or stalls.
  - Any exception anywhere -> programmatic fallback. The programmatic agent
    is therefore a hard floor: the LLM can only add value.
"""
from __future__ import annotations
import re
import sys
import time
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

# programmatic fallback (the proven v79 agent, imported as a plain class)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "prog_agent", str(_HERE.parent / "my_agent.py"))
_prog = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_prog)
ProgAgent = _prog.MyAgent


SYSTEM_PROMPT = (
    "You are an agent playing a 64x64 grid puzzle game. "
    "Each turn you see the board as objects plus an ASCII color grid, the "
    "valid actions, and recent history. Decide the next 1-6 actions that "
    "best advance toward clearing the level, preferring the FEWEST actions. "
    "Actions: ACTION1-5 and ACTION7 are simple key presses (movement / "
    "interact; their meaning varies per game). ACTION6 is a mouse click "
    "needing coordinates. RESET restarts the level.\n"
    "Reply in EXACTLY this format:\n"
    "PLAN: <one short line of reasoning>\n"
    "ACTIONS: <comma-separated, e.g. ACTION1, ACTION1, ACTION6 12 30>\n"
    "Use 'ACTION6 <row> <col>' for clicks. Output nothing after ACTIONS."
)

_ACT_RE = re.compile(r"ACTION\s*([1-7])(?:\s+(\d+)\s+(\d+))?", re.I)
_RESET_RE = re.compile(r"\bRESET\b", re.I)


class LLMAgent(Agent):
    MAX_ACTIONS = 16000
    LLM_SEQ_MAX = 6          # max actions taken from one LLM reply
    LLM_STALL = 40           # LLM actions w/o score-up -> hand to fallback
    FALLBACK_STINT = 60      # fallback actions before retrying the LLM
    LLM_FAIL_CAP = 4         # consecutive LLM failures -> fallback for good

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.prog = ProgAgent(*a, **k)          # warm world model
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
        self.fallback_until = 0
        self.mode = "llm" if self.client is not None else "prog"
        self._t0: Optional[float] = None

    # ---- framework contract ------------------------------------------
    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        self.n += 1
        if self._t0 is None:
            self._t0 = time.time()
        # keep the programmatic world model warm on EVERY frame
        try:
            prog_choice = self.prog.choose_action(frames, latest_frame)
        except Exception:                        # noqa: BLE001
            prog_choice = None

        grid = self._grid(latest_frame)
        score = getattr(latest_frame, "levels_completed", 0) or 0
        if score > self.prev_score:
            self.last_scoreup = self.n
            self.queue.clear()                   # new level: replan
        self.prev_score = score

        # hard fallback conditions
        if self.mode == "prog" or self.client is None \
                or self.llm_fails >= self.LLM_FAIL_CAP \
                or self.n < self.fallback_until:
            return prog_choice if prog_choice is not None \
                else self._safe_default(latest_frame)

        # LLM stalled? hand to fallback for a stint
        if self.n - self.last_scoreup > self.LLM_STALL:
            self.fallback_until = self.n + self.FALLBACK_STINT
            self.last_scoreup = self.n           # reset stall clock
            return prog_choice if prog_choice is not None \
                else self._safe_default(latest_frame)

        # execute queued LLM actions
        avail = self._avail(latest_frame)
        if not self.queue:
            self._query_llm(grid, avail, latest_frame)
        while self.queue:
            act = self.queue.popleft()
            if any(a.name == act.name for a in avail):
                self.history.append(act.name)
                self.prev_grid = grid
                return act
        # queue empty/illegal -> fallback this turn
        return prog_choice if prog_choice is not None \
            else self._safe_default(latest_frame)

    # ---- LLM query ---------------------------------------------------
    def _query_llm(self, grid, avail, frame) -> None:
        try:
            last_change = None
            if self.prev_grid and grid and len(self.prev_grid) == len(grid):
                last_change = sum(
                    1 for r0, r1 in zip(self.prev_grid, grid)
                    for a, b in zip(r0, r1) if a != b)
            names = [a.name + ("(row col)" if a.is_complex() else "")
                     for a in avail]
            obs = _render.render_observation(grid, names, self.history,
                                             last_change)
            reply = self.client.chat(SYSTEM_PROMPT, obs, max_tokens=256)
            self.queue = self._parse(reply, avail)
            self.llm_fails = 0 if self.queue else self.llm_fails + 1
        except Exception:                        # noqa: BLE001
            self.llm_fails += 1
            self.queue = deque()

    def _parse(self, reply: str, avail) -> deque:
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
                    row = max(0, min(63, row))
                    col = max(0, min(63, col))
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
        if not g:
            return []
        return [list(r) for r in g[-1]]

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
