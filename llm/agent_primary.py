"""LLM-PRIMARY agent for ARC-AGI-3 (Duck-style).

Inverted vs the rescue design (agent.py):
  - The LLM DRIVES from turn 1. It reasons about the board every decision
    point and acts deliberately, aiming to clear each level in as few
    actions as possible. The prize metric is efficiency-SQUARED, so a level
    solved in 40 moves scores ~100x one solved in 400.
  - Each LLM call yields a short PLAN (up to LLM_SEQ_MAX actions), often via
    a fenced python block that runs BFS/greedy search in a sandbox. One call
    -> many actions keeps the number of (expensive) model calls bounded while
    the actions themselves stay algorithmically tight.
  - The programmatic agent (v79) is kept WARM every frame (it learns from the
    observed trajectory regardless of who chose the action) and serves as the
    per-frame floor: any LLM error / empty / illegal result defers to it.
  - When the per-game LLM-call budget is spent, control passes to the floor
    for the rest of the game. This is deliberate: the LLM spends its scarce,
    efficient actions on the levels it can crack fast; brute-force then mops
    up the remaining (already-lost-on-efficiency) levels for free, and can
    only ADD levels. The agent can never score below the v79 floor.

Backend: set ARC_LLM_BACKEND=openai and point ARC_LLM_BASE_URL at a vLLM
OpenAI-compatible server (concurrent requests are batched on-GPU). The
serializing lock in llm_client is bypassed for the openai backend so the
110 concurrent games actually batch.
"""
from __future__ import annotations
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

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

_MODULE_START = time.time()


SYSTEM_PROMPT = (
    "You are playing an unfamiliar 64x64 grid puzzle game, from the start, in "
    "full control.\n"
    "SCORING: (human_moves / your_moves) squared, awarded ONLY for levels you "
    "actually CLEAR. A level you never clear scores 0 no matter how many "
    "actions you spend on it. So: exploring is nearly free on early levels; "
    "NEVER give up on a level -- clearing it slowly still beats not clearing "
    "it. Once you understand a level, execute crisply to keep moves low.\n"
    "COMMON WIN CONDITION (check this FIRST): most of these games are a COVER "
    "task -- every MOVER object must end up on top of a DESTINATION. The "
    "destinations are ALREADY on the board from the very first frame: look for "
    "a group of 2+ objects that are identical (same shape+color), small, and "
    "STATIC (don't move when you act). Your job becomes: find which action "
    "moves a mover, then move each mover onto a destination. Variants: remove "
    "all objects of one color; make a region match a target pattern; cover the "
    "destinations in a specific order. Reframe from 'what does this button do' "
    "to 'get each mover onto a destination'.\n"
    "You see the board as connected same-color OBJECTS (with a shape signature "
    "so you can spot identical shapes) plus an ASCII color grid, the VALID "
    "ACTIONS, RECENT ACTIONS, and WHAT YOU'VE LEARNED so far.\n"
    "FEEDBACK each turn: how many cells your last action changed (0 = nothing), "
    "your last click's coords + effect, and lists of USELESS actions / DEAD "
    "cells. Never repeat something the feedback says did nothing.\n"
    "MOUSE (ACTION6 row col) clicks one of 4096 cells. The real controls are "
    "usually only a FEW scattered cells -- do NOT sample the board evenly. "
    "Click DELIBERATELY on object centers and small isolated markers. A cell "
    "you've clicked with no effect is dead; never click it again.\n"
    "How to play well:\n"
    "- First moves: probe efficiently -- try EACH movement action once and a "
    "few DIFFERENT deliberate clicks in ONE plan, then read which did "
    "something. Don't repeat a 0-change action/cell.\n"
    "- Then pursue the cover hypothesis directly. Do not wander, and do not "
    "click the same object over and over.\n"
    "- A long thin strip flush against an edge is usually a timer/HUD bar, "
    "NOT clickable pieces. Never click along it segment by segment.\n"
    "- When the goal cell is known but the path isn't, SEARCH: write code that "
    "computes the move sequence.\n"
    "MEMORY -- most important: end EVERY reply with a line:\n"
    "LEARNED: <one short sentence stating any NEW mechanic you just confirmed, "
    "e.g. 'ACTION2 moves the red block down' or 'clicking a small green marker "
    "teleports the mover there'. Say 'none' if nothing new.>\n"
    "These notes are the ONLY thing that carries across turns and levels, so "
    "keep them accurate and about MECHANICS (not this level's exact layout, "
    "which changes next level).\n"
    "Reply with EXACTLY ONE of these two action formats, then the LEARNED "
    "line:\n"
    "1) Direct:\n"
    "PLAN: <one short line of intent>\n"
    "ACTIONS: <comma list, e.g. ACTION2, ACTION2, ACTION6 12 30>\n"
    "   (ACTION6 is a click and needs 'ACTION6 <row> <col>'.)\n"
    "2) Code (prefer when search/computation helps): a single fenced python "
    "block. Bound names: `grid` (list[list[int]] rows), `objects` (dicts with "
    "color,n,center=[row,col],bbox,shape), `valid` (action-name strings), "
    "`banned_cells` (set of dead (row,col) that do nothing -- skip them), "
    "`clicked` (set of already-clicked (row,col)). You MUST assign a list "
    "`plan`; each item is a move string like 'ACTION2' or a click ('ACTION6', "
    "row, col). Example:\n"
    "```python\n"
    "# click the smallest untried object\n"
    "cand = [o for o in objects if tuple(o['center']) not in banned_cells]\n"
    "o = min(cand, key=lambda o: o['n'])\n"
    "plan = [('ACTION6', o['center'][0], o['center'][1])]\n"
    "```\n"
    "Re-clicking a cell that KEEPS changing the board is GOOD -- that is how "
    "you push a mover step by step onto its destination; only avoid cells in "
    "banned_cells. Keep each plan <= 8 actions so you can watch the result."
)

_ACT_RE = re.compile(r"ACTION\s*([1-7])(?:\s+(\d+)\s+(\d+))?", re.I)
_RESET_RE = re.compile(r"\bRESET\b", re.I)
_CODE_RE = re.compile(r"```(?:python)?\s*(.+?)```", re.S | re.I)
_LEARNED_RE = re.compile(r"LEARNED:\s*(.+)", re.I)

_SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "abs": abs,
    "sorted": sorted, "sum": sum, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "any": any, "all": all, "int": int, "float": float,
    "bool": bool, "str": str, "round": round, "reversed": reversed,
    "divmod": divmod, "sorted": sorted, "frozenset": frozenset,
}


def _envint(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:                            # noqa: BLE001
        return default


class LLMPrimaryAgent(Agent):
    MAX_ACTIONS = 16000
    LLM_SEQ_MAX = 8            # actions taken from a single LLM reply
    # per-game LLM-call budget (env-overridable so Colab can dial it down)
    MAX_LLM_CALLS = _envint("ARC_LLM_MAX_CALLS", 300)
    LLM_FAIL_CAP = _envint("ARC_LLM_FAIL_CAP", 12)   # consecutive fails -> floor
    # actions the FLOOR drives at the START of each level before the LLM
    # engages. This protects the floor's cheap/lucky fast wins (ar25 lv0 in
    # 74, vc33 lv0 in 5) -> LLM-primary is then >= floor on easy levels. The
    # LLM only takes levels the floor hasn't cracked in this window (which
    # were scoring ~0 on efficiency anyway). 100 protects the floor's cheap
    # wins (ar25 lv0@74, lv1@+52); set 0 to probe raw LLM capability.
    LEVEL_FLOOR_OPENING = _envint("ARC_LLM_FLOOR_OPENING", 100)
    # stop LLM-driving a level after this many calls with no level-up; the
    # floor brute-forces that level while the LLM stays ready for the next one
    STUCK_CALLS = _envint("ARC_LLM_STUCK_CALLS", 40)
    GLOBAL_DEADLINE_S = _envint("ARC_LLM_DEADLINE_S", 25200)  # 7h rerun safety
    MAX_TOKENS = _envint("ARC_LLM_MAX_TOKENS", 512)

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
        self.n = 0                    # total frames
        self.consec_fails = 0
        self.n_llm_calls = 0          # total LLM calls this game
        self.calls_this_level = 0     # LLM calls since last level-up
        self.level_start_n = 0        # frame index at the last level-up
        self.last_change = None
        self.last_act_name = None     # action executed on the previous frame
        self.last_click_rc = None     # (row,col) of the last click executed
        self.dead_counts = {}         # simple-action name -> consec no-change
        self.click_effect = {}        # (row,col) -> last cells-changed by a click
        self.click_count = {}         # (row,col) -> times clicked
        self._banned_cells = set()    # cells the LLM may not click any more
        self.world_notes = []         # learned MECHANICS (persist across levels)
        self.last_change_bbox = None  # region of the last action's change
        self.grid_at_query = None     # board snapshot at the last LLM query
        self.floor_only = False       # latched once we hand the game to prog

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    # ---- main loop ---------------------------------------------------
    def choose_action(self, frames, latest_frame) -> GameAction:
        self.n += 1
        # keep the floor warm (it learns from the observed trajectory and is
        # our fallback). Its own action_count/telemetry advance here too.
        try:
            prog_choice = self.prog.choose_action(frames, latest_frame)
        except Exception:                        # noqa: BLE001
            prog_choice = None

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            # let the floor own reset/retry bookkeeping
            self.queue.clear()
            return self._ret(prog_choice if prog_choice is not None
                             else GameAction.RESET)

        grid = self._grid(latest_frame)
        # measure what our previous action did -- count AND where (bbox), so
        # the model can infer WHAT moved and in which direction
        if self.prev_grid and grid and len(self.prev_grid) == len(grid):
            changed = [(r, c)
                       for r, (r0, r1) in enumerate(zip(self.prev_grid, grid))
                       for c, (a, b) in enumerate(zip(r0, r1)) if a != b]
            self.last_change = len(changed)
            if changed:
                rs = [p[0] for p in changed]; cs = [p[1] for p in changed]
                self.last_change_bbox = (min(rs), min(cs), max(rs), max(cs))
            else:
                self.last_change_bbox = None
        self.prev_grid = grid
        # attribute the change to the previously-executed action; a simple
        # (non-click) action that repeatedly changes nothing is inert here
        if self.last_act_name and self.last_change is not None:
            try:
                cplx = GameAction[self.last_act_name].is_complex()
            except Exception:                    # noqa: BLE001
                cplx = True
            if not cplx:
                if self.last_change == 0:
                    self.dead_counts[self.last_act_name] = \
                        self.dead_counts.get(self.last_act_name, 0) + 1
                else:
                    self.dead_counts[self.last_act_name] = 0
            elif self.last_click_rc and self.last_click_rc[0] is not None:
                # remember what clicking THIS cell did (so we can tell the LLM
                # and stop it re-clicking dead / over-clicked cells)
                self.click_effect[self.last_click_rc] = self.last_change
                self.click_count[self.last_click_rc] = \
                    self.click_count.get(self.last_click_rc, 0) + 1

        score = getattr(latest_frame, "levels_completed", 0) or 0
        if score > self.prev_score:
            self.prev_score = score
            self.queue.clear()            # new level: replan from scratch
            self.calls_this_level = 0
            self.consec_fails = 0
            self.level_start_n = self.n   # floor gets a fresh opening window
        # a level-up also re-arms the LLM if the floor had been handed a
        # stuck level (the next level may well be LLM-solvable)

        avail = self._avail(latest_frame)

        want_llm = (
            self.client is not None
            and not self.floor_only
            and self.n_llm_calls < self.MAX_LLM_CALLS
            and self.consec_fails < self.LLM_FAIL_CAP
            and self.calls_this_level < self.STUCK_CALLS
            # let the floor open each level; LLM engages only if the floor
            # hasn't cleared it within the opening window
            and (self.n - self.level_start_n) >= self.LEVEL_FLOOR_OPENING
            and time.time() - _MODULE_START < self.GLOBAL_DEADLINE_S
        )
        if self.n_llm_calls >= self.MAX_LLM_CALLS \
                or (self.client is None):
            self.floor_only = True

        if want_llm:
            if not self.queue:
                # did our previous plan change anything? (stuck detector)
                stuck = (self.grid_at_query is not None
                         and grid == self.grid_at_query)
                dead = sorted(n for n, c in self.dead_counts.items() if c >= 3)
                self.grid_at_query = grid
                self._query_llm(grid, avail, dead, stuck)
            while self.queue:
                act = self.queue.popleft()
                if any(x.name == act.name for x in avail):
                    self.history.append(act.name)
                    return self._ret(act)
            # LLM produced nothing usable this frame -> floor for this frame

        return self._ret(prog_choice if prog_choice is not None
                         else self._safe_default(latest_frame))

    def _absorb_learned(self, reply) -> None:
        # capture the model's LEARNED: line -> persistent mechanics memory.
        # This is the cross-level knowledge the naive agent lacked; it is NOT
        # cleared on level-up (mechanics carry over even when layout resets).
        m = _LEARNED_RE.search(reply or "")
        if not m:
            return
        note = " ".join(m.group(1).split())[:160].strip(" .")
        if not note or note.lower() in ("none", "n/a", "nothing", "nothing new"):
            return
        low = note.lower()
        if any(low == n.lower() for n in self.world_notes):
            return
        self.world_notes.append(note)
        if len(self.world_notes) > 12:          # keep the most recent mechanics
            self.world_notes = self.world_notes[-12:]

    def _ret(self, action):
        self.last_act_name = getattr(action, "name", None)
        self.last_click_rc = None
        try:
            if action.is_complex():
                ad = getattr(action, "action_data", None)
                if ad is not None:
                    self.last_click_rc = (getattr(ad, "y", None),
                                          getattr(ad, "x", None))
        except Exception:                        # noqa: BLE001
            pass
        return action

    # ---- LLM query ---------------------------------------------------
    def _query_llm(self, grid, avail, dead=(), stuck=False) -> None:
        try:
            self.n_llm_calls += 1
            self.calls_this_level += 1
            names = [a.name + ("(row col)" if a.is_complex() else "")
                     for a in avail]
            notes = [f"CURRENT LEVEL {self.prev_score}. "
                     f"LLM calls on this level: {self.calls_this_level}."]
            if self.last_act_name == "ACTION6" and self.last_click_rc \
                    and self.last_click_rc[0] is not None:
                r, c = self.last_click_rc
                notes.append(f"Your last click was ({r},{c}); it changed "
                             f"{self.last_change} cells.")
            if self.last_change and self.last_change_bbox:
                r1, c1, r2, c2 = self.last_change_bbox
                notes.append(f"Those changes were in the region rows {r1}-{r2}, "
                             f"cols {c1}-{c2} -- compare to your action to infer "
                             f"WHICH object moved and in which direction.")
            if stuck:
                notes.append("!! Your LAST plan changed the board by 0 cells "
                             "-- it did NOTHING. Do something DIFFERENT: a "
                             "different action or a click on a DIFFERENT cell. "
                             "Do NOT repeat it.")
            if dead:
                notes.append("USELESS actions (changed 0 cells every time) -- "
                             "NEVER pick these: " + ", ".join(dead))
            # ban ONLY dead cells (clicking them changed 0 cells). Do NOT ban
            # productive cells by click-count: re-clicking a cell that keeps
            # changing the board is how you push a mover step by step toward
            # its destination -- banning those blocks the actual solution.
            self._banned_cells = {rc for rc, ch in self.click_effect.items()
                                  if ch == 0}
            if self._banned_cells:
                banned = sorted(self._banned_cells)
                shown = ", ".join(f"({r},{c})" for r, c in banned[:16])
                more = "" if len(banned) <= 16 \
                    else f" (+{len(banned) - 16} more)"
                notes.append(f"DEAD cells (clicking changed 0 cells) -- clicks "
                             f"on these are IGNORED, pick others: {shown}{more}")
            if self.world_notes:
                notes.append("WHAT YOU'VE LEARNED (mechanics, persists across "
                             "levels):\n- " + "\n- ".join(self.world_notes))
            obs = "\n".join(notes) + "\n" + _render.render_observation(
                grid, names, self.history, self.last_change)
            reply = self.client.chat(SYSTEM_PROMPT, obs,
                                     max_tokens=self.MAX_TOKENS)
            self._absorb_learned(reply)
            q = self._from_code(reply, grid, avail)
            if not q:
                q = self._parse(reply, avail)
            self.queue = q
            self.consec_fails = 0 if q else self.consec_fails + 1
            if os.environ.get("ARC_LLM_DEBUG"):
                snippet = " ".join(reply.split())[:220]
                print(f"[LLMq lvl={self.prev_score} call#{self.n_llm_calls} "
                      f"-> {len(q)} acts] {snippet}", flush=True)
        except Exception:                        # noqa: BLE001
            self.consec_fails += 1
            self.queue = deque()

    def _from_code(self, reply, grid, avail) -> deque:
        m = _CODE_RE.search(reply)
        if not m:
            return deque()
        code = m.group(1)
        objs = _render.segment(grid)
        # the model routinely writes `... not in banned_cells` / `BANNED`;
        # provide them (a set of dead (row,col)) so its filter code runs
        # instead of raising NameError -> empty plan -> wasted call
        banned = set(self._banned_cells)
        env = {"grid": grid, "objects": objs,
               "valid": [a.name for a in avail], "plan": [],
               "banned_cells": banned, "BANNED": banned, "dead_cells": banned,
               "clicked": set(self.click_effect.keys())}
        # the model often writes bare ACTION1 / ACTION6(r,c) in code (not
        # quoted) -> NameError. Bind them: moves as strings, the click as a
        # callable so ACTION6(r,c) and ('ACTION6',r,c) both work.
        for _i in (1, 2, 3, 4, 5, 7):
            env[f"ACTION{_i}"] = f"ACTION{_i}"
        env["ACTION6"] = lambda r, c: ("ACTION6", r, c)
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
                if (row, col) in self._banned_cells:
                    continue                     # hard-drop banned clicks
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
                    if (row, col) in self._banned_cells:
                        continue                 # hard-drop banned clicks
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


MyAgent = LLMPrimaryAgent
