# ARC-AGI-3 project — handoff notes for Claude

**If you are Claude starting a fresh session: read this fully, then continue
the work. The user (bader) is competing in ARC Prize 2026 - ARC-AGI-3 on
Kaggle (username: baderalotaibi11).**

## Current state (as of 2026-08-21, evening session — Windows native now)

- Games cache MOVED: it is now `Downloads/kaggle/arc_games` (inside this
  folder, not `Downloads/arc_games`).
- **Windows host has Python 3.13 → `pip install arc-agi` works DIRECTLY.**
  No typing_extensions hack needed. Forget the sandbox 3.10 workaround
  unless running in a Linux sandbox again.
- `eval_harness.py` (this folder) = the local evaluator. Usage:
  `python eval_harness.py --all --agent my_agent_vXX.py` (parallel,
  ~4 min for 25 games) or `--game ls20`. Official
  EnvironmentScoreCalculator scoring (level-index weighted, efficiency^2,
  1-based level weights). Saves eval_results.json.
- `diag.py <game> --agent <file>` = plays one game in-process and dumps
  agent internals (states, avatar color, dirmap, reasons, action counts).
- `my_agent.py` = current champion. Versioned copies: `my_agent_v10.py`
  (old champion), `my_agent_v11.py`, `my_agent_v12.py`, ...
- First Kaggle submission (v1) scored 0.00 (succeeded, no crash).
  1 submission allowed per day.

## Score history (LOCAL harness on this Windows machine, 25 games, 4000 act.)

| Version | Levels | Mean score (0-100) | Note |
|---|---|---|---|
| v10 | 4 | 0.0036 | ar25 g50t lf52 sp80, 1 each |
| v11 | 7 | 0.0124 | + volatility-masked hashing, wall colors, click tiers (lp85 2, m0r0, vc33; lost lf52) |
| v12 | 7 | 0.0125 | + diff-based avatar detection (bbox-restricted), component locate |
| v13 pre-fix | 12 | 0.0492 | + interface learning, row/col volatility mask, avatar ignores mask |
| v13 | 22 | 0.1911 | + **GameAction.from_id FIX** (see below) — 17 games ≥1 level |
| v14 | 22 | 0.1911 | + replay pruning (identical locally; better after deaths) |
| v15 | 19 | 0.1954 | + autonomous-motion filter, adaptive vol pressure, undo detect, novelty fallback — REGRESSED levels |
| v16 | 23/21/17 (salts 0/1/2) | 0.19/0.06/0.09 | v15 features gated to never-scored games + opening probe + 1-3-cell step dirmap + body-overlap nav targets |
| v17 | 21/20/15 | ~same | + rigid-translation avatar votes, pos-from-motion, avatar-aware wall blame — worse on levels |
| v18 | 16/14/14 = 44 | lower | + exact-target nav + deterministic click salience — REGRESSED HARD (click order determinism killed cross-state coverage; lp85 5->2, s5i5 lost) |
| v19/v20 | 44 | — | rotation + partial fixes; patches half-failed (see encoding trap) — still 44 |
| v21 | 16 (salt0) | 0.0598 | all v17-v20 features verified present — regression is systematic in the v17+ line |
| v22 | 23/23/16 = 62 | 0.191/0.068/0.087 | **RESTART FROM v16** + ratio-useless + unique-delta — beat v16 |
| v23 | 22/22/17 = 61 | 0.209/0.079/0.104 | v22 + tracking fixes (pos-from-motion, motion-gated wall blame) — means up, levels -1 |
| v24 | 21/21/17 = 59 | — | v23 + step cap 5 — REJECTED (noise votes) |
| v25 | 19/21/20 = 60 | 0.205/0.096/0.117 | v23 + goal-color click tier — means best, levels -2 |
| v27 | 22/22/18 = 62 | 0.211/0.081/0.107 | v22 + pushable-blocks-not-walls — ties v22 levels, beats mean on all salts |
| v28 | 22/22/17 = 61 | ~v23 | v27 + v23 tracking fixes — rejected (tracking keeps costing 1 level) |
| v29 | 19/21/21 = 61 | 0.208/0.098/0.120 | v27 + goal-color clicks — best mean avg, levels -1, rejected |
| v30 | 17/16/14 = 47 | — | v27 + unbounded symmetry clicks — REJECTED (floods click queue) |
| v31 | 22/22/18 = 62 | =v27 | v27 + symmetry clicks CAPPED at <=8 broken pairs (fires rarely) |
| v32 | 23/22/16 = 61 | ~v27 | v31-base + nav-arrival click — rejected |
| v33 | 22/22/18 = 62 | =v31 | + strict rigid votes for UI-shared colors — literally never fired |
| v34 | 21/21/17 = 59 | 0.252/0.058/0.099 | + rigid-gated 4-6-cell steps (ls20 avatar moves 5/press!), adaptive radius — best salt0 mean ever, levels -3, rejected |
| v35 | 19/21/21 = 61 | 0.208/0.098/0.120 | v31 + goal-color click tier — broadest capability set |
| v36 | 16/20/17 = 53 | 0.200/0.592/0.101 | RESEED 700 — salt1 jackpot: **lp85 = 13.90 (5 levels, L5 in 42 actions vs baseline 41!)** but levels down, rejected |
| v37 | 17/19/19 = 55 | — | clicks/state 40 — rejected |
| v38 | 18/7/18 = 43 | — | weighted click sampling — rejected hard |
| v39 | 19/21/21 = 61 | 0.208/0.098/0.120 | v35 + USELESS_AFTER_SIMPLE 15 (faster interface convergence, same local results) |
| v40 | =v39 | =v39 | STUCK_WINDOW 100 — neutral, dropped |
| v41 | 17/17/18 = 52 | — | adaptive reseed pacing — REJECTED (reseed axis is dead) |
| **v42 = champion (pushed)** | 19/21/21 = 61 | =v39 | + plans/replays/momentum cleared on level-up (identical locally, safer semantics) |
| v43 | 19/20/19 = 58 | 0.269/0.100/0.117 | + per-level dead-cell click memory — best avg mean (0.162) but levels -3 SYSTEMATICALLY (state-dependent buttons banned too early); shelved, revisit with state-conditional banning |
| v44 | 19/19/19 = 57 | 0.214/0.078/0.114 | v43 threshold 6 — worse both axes |
| v45 | 18/20/21 = 59 | 0.206/0.100/**0.410** = avg 0.239 | CONTEXTUAL dead-click rules keyed (x, y, shown color) — mean +68%; lp85 salt2 = 8.70 (L1 28 acts vs baseline 17, L5 56 vs 41). First mechanism whose mean gain comes from the intended causal path. |
| v46 | 16/17/18 = 51 | avg 0.198 | + precision tier0 + repeat-click — REJECTED |
| v47 | 18/17/12 = 47 | avg 0.214 | precision tier0 alone — REJECTED (any click-order de-randomization costs levels; the shuffle earns its keep) |
| v48 | 18/22/22 = **62** | 0.206/0.101/0.410 = avg 0.239 | v45 + click-only per-state cap 96→192 (deeper sweeps pay once dead-context pruning cleans the lists) — best of both axes |
| v49 | 19/18/20 = 57 | avg 0.239 | mixed-game cap 48 — REJECTED (keyboard budget crowded out) |
| v50 | =v48 | =v48 | click-only cap 288 — never binds past 192, dropped |
| **v51 = champion (kernel v7 carries v48≡v51)** | =v48 | =v48 | + fallback clicks skip dead contexts (identical locally, better prior) |

| v52 | 16/17/14 = 47 | — | order-seeking click preference — REJECTED (4th confirmation: ANY change to the v45/v48 click ordering loses levels; it is a local optimum) |
| v53 | 17/22/22 = 61 | avg 0.232 | BFS plan depth 50 — rejected (salt0 dip) |

True click productivity (masked, v51): 64-89% across click games —
the dead-click problem is SOLVED. tn36 makes 2,241 productive clicks
with 0 levels: the frontier is GOAL INFERENCE for toggle-lattice
puzzles (template matching between board regions), not click hygiene.
NB: raw-frame 'productivity' measurements are poisoned by counter
pixels — always use the agent's masked-state counters.

REPLAY MINING STATUS (2026-08-21 late): user connected the Kaggle MCP
connector on claude.ai, but connector auth binds at SESSION START -- calls
from the then-running session stayed Unauthenticated. RETRY
list_submission_episodes({submissionId: ...}) IN A FRESH SESSION. Note:
the unauthenticated public endpoint returned zero episodes for the
submission (private eval games may not expose episodes at all -- verify
once authenticated before assuming replays exist).

Repo layout: agent versions live in versions/, gameplay GIFs in media/
(make_gif.py renders them; README links them; build_notebook.py strips
the gallery from the notebook embed since Kaggle has no media files).

LB fact: submission 2026-08-21 01:58 (v9 agent!) scored 0.17 public.
Daily quota resets 00:00 UTC — submit kernel v7 (v48) after reset.

| **v54 = champion (kernel v8, SUBMIT THIS)** | =v51 (verified salt0) | =v51 | + rerun TELEMETRY: [MYA] lines in the submission log = diagnostics from the 110 private games |
| v55 | 18/21/21 = 60 | avg 0.240 | copy-task two-region mismatch tier -- SHELVED (never fires locally, -2 levels; revisit if telemetry shows copy tasks) |

| v56 | 16/20/22 = 58 | avg 0.243 | react-interact + small-object nav + overlap-arrival -- REJECTED (nav retarget displaces luck) |
| v57 | 17/12/23 = 52 | -- | react-interact alone -- REJECTED (salt1 collapse) |

**wa30 DECODED (via PNG snapshots + scripted play)**: flying avatar
(brown, color 14, 3-cell steps), three 12-cell yellow targets. Mechanic:
targets are SOLID except from below; ENTER one from below -> it
highlights green (color 3) -> press ACTION5 while highlighted -> target
consumed. Collect all three (per level, presumably) to score. The stock
agent collects at most one by luck. Needed capabilities (future):
approach-direction learning (retry blocked targets from other sides) and
persistent multi-target task memory. Snapshot tool: snapshots.py; the
mechanisms tried (react-interact, small-object rare rule, overlap
arrival) all FAILED multi-seed -- keep the insight, not the code.

**ZERO-GAME CENSUS (visual, via snapshots.py)** — capability wish-list,
to be prioritized by private-set telemetry:
1. INSTRUCTIONS-ON-SCREEN family: sk48 (bottom legend shows required
   visit ORDER for a path-drawing cursor), dc22 (right half shows outlined
   template shapes to build from left-half pieces), tn36 (0/1 indicator
   strip + confirm button). Needs: read goal from a screen region,
   execute. Biggest family.
2. COLLECT-WITH-INTERACTION: wa30 (enter target from below -> highlight
   -> A5). Needs approach-direction learning + interact-while-highlighted.
3. LOCK-AND-KEY: ka59 (two rooms, dark door in corridor, yellow goal
   pads), ls20 deeper levels. Needs multi-stage objectives.
4. Others: re86 multi-cursor alignment; sc25 gravity; tr87 unknown
   (283-state graph fully explored, no score); lf52/sb26/su15 click
   sequences.

AUTO-SUBMIT: Windows scheduled task "KaggleArcSubmit" fires 03:01 local
(00:01 UTC Aug 22) running submit_kernel.cmd -> submits kernel v8 via CLI
(PC must be ON). A persistent monitor polls for the new submission+score.
LEADERBOARD CONTEXT (2026-08-21): #1 = 3.57. User target: LB 10.

RERUN LOGS ARE HIDDEN for this comp (user verified: only the 28s commit
log is visible). The DAILY LB SCORE is the only private-set signal ->
every submission = one controlled experiment. Calibration: LB ~= 1.8 x
local_mean (v54: local 0.142 -> LB 0.26). Targets: LB 4 = local ~2.2;
LB 10 = local ~5.5.

| **v58 = champion (kernel v10, daily submit)** | 19/22/22 = **63** | =v54 means | DESPERATION MODE: score==0 && actions>1200 unlocks gated arsenal (react-interact, small-object nav, overlap arrival, copy-task tier); scored games byte-identical. + 8000-action budget wall-clock-capped at 160s/game. THE pattern for risk-free capability adds. |

| v60 | 57 / 60 gated-push variants | -- | Sokoban-slide push (ka59 archetype) -- REJECTED both variants: even desperation-mode capabilities compete for zero-game recovery budget |
| **v61 = champion (kernel v12, daily submit)** | 19/22/22=63 @4k; 23/22/23=68 @8k; 25 @12k salt0 | ~same means | v59 + MAX_ACTIONS 16000 (the 160s/game wall cap governs on Kaggle; total <= 4.9h for 110 games). BUDGET CURVE: 4k->63, 8k->68, 12k->75-ish levels -- budget was binding after all at v59 capability level. |

CERTIFICATION PROTOCOL: iterate at 4k (fast), certify champions at 8k+
(matches kernel behavior). ka59 = sliding-push archetype (bump item ->
slides to obstruction; goal = push onto hollow-square pad) -- documented,
capability parked after two failed gated attempts.

AUDIT (2026-08-22, 7-angle multi-agent review): 10 verified findings.
CRITICAL: (1) plan/plan_expected OFF-BY-ONE -- every multi-step plan since
v3 aborted after its first action; (2) mask_rev in state hashes orphans
replay bank + graph on every mask recompute; (3) empty available_actions
frame permanently pollutes simple_seen (phantom actions); (4) falsy-zero
`color_at or -1` excludes color-0 objects from click tiers; plus replay
livelock, death-frame vol pollution, wall-bump uselessness, corridor
self-masking, desperation gate never reopening after a score, ~2x
redundant grid scans. ALL fixes implemented in versions/my_agent_v64
(all 10), v65 (correctness only), v66 (trajectory-safe only) -- BUT
v64=51, v65=54, v66=52 vs champion 63 @4k: even pure fixes shift the
luck-coupled trajectories net-negative locally. Champion stays v62.
LESSON: the 25-game 3-salt metric is saturated by variance (+-5); fixing
real bugs reads as regression. Next arc needs either many more salts for
statistical power, or the planner fix paired with better plan TARGETS
(frontier-chasing may be the true weakness the broken planner was hiding).

LB RESULT 2026-08-23: v67 (16k budget + arsenal + legend + speed cache)
scored 0.26 -- EXACTLY FLAT vs v54. Calibration k ~= 1.1 now (local mean
~0.24 -> LB 0.26). VERDICT: budget/levels are a dead end for score; the
(b/a)^2 square zeroes slow completions. ONLY near-baseline first-solves
move the LB. NEXT ARC = SOLVER MODE: recognize an archetype (collect /
push-to-pad / legend-path), exit the exploration stack, run a dedicated
tight execution loop at scripted quality (manual wa30 play hit baseline
pace with the same primitives -- composition is the gap, not capability).

SOLVER ARC (2026-08-23): v69 collect-solver = -5 levels (ungated
takeover). v70 adds the EVIDENCE GATE (world must confirm the archetype:
a target vanished during exploration, or a reaction fired near one,
before the solver commits budget) -> 20/22/20 = 62, ~zero local cost,
capability aboard. wa30 STILL unfinished after 5 attempts (3-direction
dirmap breaks the from-below route; transient self-blocks) -- wa30 is
RETIRED as validation target; solver validation = suite-level only.
TONIGHT kernel v17 = v68 planner-efficiency experiment; tomorrow:
if LB > 0.26 -> champion = v70+planner; else v70 alone as kernel v18.

TOP-3 SPEED MECHANISMS (2026-08-23, v71 = champion by MEAN-first):
1. Cross-level winning-sequence replay (one-shot attempt of level N's
   pruned win at level N+1 start) -- ar25 L2 in 52 actions vs baseline
   50 (!) = 5.67 pts; vc33/tu93/lp85 all cascade now.
2. Last-win click coordinates = supreme click tier at the new level.
3. Score-up plans exempt from the plan throttle.
v71 multi-seed: 13/19/20 = 52 levels but means 0.175/0.332/0.350 =
avg 0.286 -- BEST of campaign, and spread across salts (not one lucky
lp85). Predicted LB ~0.31. Levels-first championship is DEAD; mean-first
is the rule now (k~1.1 means local mean IS the LB).

| v71 | 20/19/20 = 59 | 0.407/0.332/0.350 = avg 0.363 | TOP-3 SPEED MECHANISMS: cross-level winning-sequence replay (one-shot at new level), last-win clicks as supreme tier, score-up plans exempt from throttle. +52% mean. (An earlier 0.175 salt0 reading was a STALE-FILE artifact.) |
| v72 | 22/20/19 = 61 | avg 0.351 | + mismatch-abort of strategy replay -- levels up, mean down, rejected |
| **v73 = champion (kernel v18, submits 2026-08-24 00:01 UTC)** | 20/22/21 = **63** | 0.406/0.338/0.339 = **avg 0.361** | + finishing-move second stage (last-20 replay) -- robustness back at zero mean cost |
| v74 | 19/22/20 = 61 | avg 0.361 | + resurrected planner -- neutral, parked |
| v75/v76 | 16/16/15 = 47 | avg 0.315 | win-click COLOR generalization (all / nearest-2) -- REJECTED, any top-tier addition costs levels |

ITERATION TEMPO (2026-08-23): local mean ~= LB (k~1.1) so iterate locally
at machine speed; nightly submission is confirmation only. ALWAYS rerun
suspicious numbers (stale-file artifacts happen when evals launch
while files are mid-edit).

Kernel-push rule refined: push a new kernel version only when champion
BEHAVIOR changes locally; git commit+push on every adoption.

## Git repo rule (STANDING)

Public repo: github.com/BDR-Pro/arc-prize-2026-arc-agi-3 (created
2026-08-21, gh CLI in WSL is authenticated as BDR-Pro). Working copy =
Downloads/kaggle (arc_games/ and scratch excluded via .gitignore).
**Every champion change: git add -A && git commit && git push** (from WSL:
`cd /mnt/c/Users/bader/Downloads/kaggle`).

**Budget profile (v42, 15 scoring games, salt0)**: explore 57.4%, momentum
14.2%, plan 11.2%, fallback 8.6%, navigate 6.7%. Click games are 90%+
explore = clicking through target lists. Efficiency mechanisms (nav,
dead-cell memory) consistently trade levels for mean — the resolution is
state-conditional rules (the hypothesis layer), not more blanket bans.

**lp85 case study (v36 salt1)**: goal-colors + good-clicks carrying across
levels solved L4 in 138 and L5 in 42 actions (weight-5 level at ~baseline
= 13.9 game points). This is the "hit 4" shape: deep levels at baseline.
The blocker is early-level solve speed cascading into budget for deep ones.

**Kaggle push DONE (2026-08-21)**: v39 pushed as kernel version 3
(https://www.kaggle.com/code/baderalotaibi11/arc-agi-3-world-model-agent),
status RUNNING. The fresh token came from the Windows user env var
KAGGLE_API_KEY and now lives in WSL ~/.kaggle/access_token (correct
location for kaggle CLI 2.2.4; KAGGLE_CONFIG_DIR + old KGAT no longer
works). When the kernel run finishes, the USER clicks "Submit to
Competition" -> submission.parquet on kaggle.com (1/day).
Old NOTES warning about the leaked KGAT token: resolved — it was rotated.

**Champion rule updated (2026-08-21)**: mean game score now carries real
signal (it IS the Kaggle LB metric); levels and mean are co-primary.
Within the noise band (±2 levels / ±0.02 mean) prefer capability breadth.

**Nav paradox**: every nav-strengthening change (v23 tracking, v34 big
steps) raises means but costs 1-3 levels — nav displaces the "productive
stumbling" that momentum/explore provide. Nav's target model (rare colors
/ goal colors / frontier) is right often enough for efficiency but not to
replace exploration. Next frontier: better goal inference, not more nav.

**ls20 ground truth**: avatar = color 9, 5x3 block, moves FIVE cells per
press (perfect rigid translation); carried key = color 12. Early "2-cell
step" readings were a different phase. Delta caps must handle 5.

Budget test: doubling to 8000 actions flips almost nothing (only sc25)
— zero games are capability-limited, not budget-limited.
Scoring check: per-level actions are CUMULATIVE from previous completion
(scorecard actions_by_level differences) — no replay/reset efficiency
exploit exists; first-solve speed is the only efficiency lever.

**Multi-seed eval is mandatory**: single-run comparisons swing ±3 levels on
seed luck. `--salt N` on eval_harness varies the agent rng (game_id salt).
Compare SUM of levels over salts 0,1,2. Benchmark: **v16 = 61** (23/21/17).

**Iteration discipline (learned the hard way)**: ONE change class per
version, multi-seed evaluated, before stacking the next. The v17-v21 batch
of 5 "obviously good" mechanisms lost 17 levels net.

**Windows encoding trap**: NEVER patch agent files with python
replace-scripts (default cp1252 read/write): patterns containing em-dashes
fail SILENTLY. Use the Claude Edit tool or add encoding='utf-8' + asserts.

**Game facts from probes** (all confirmed by frame diffs):
- Most games tick a counter pixel on EVERY action (row 63 / row 0 strip)
  — poisons any "did the action do something" logic that isn't ratio-based.
- ls20: avatar color 9 (or 5?), 2-cell steps; color-9 UI icon bottom-left
  traps naive locate(); energy bar rows at bottom.
- re86: 3 crosshairs (color 9, intersections color 0) move together in
  3-cell steps; ACTION5 stamps; likely needs multi-cross alignment.
- sc25: piece falls (gravity); 4-cell horizontal steps; two actions can
  both appear to map "down" — needs duplicate-delta filtering.
- wa30: avatar 3-cell steps, dirmap forms, nav runs 880/1200 actions but
  no level — objective is not rare-color-touch.
- lf52: keyboard actions are pure no-ops (click game); only the counter
  pixel changes. tn36: clicking empty space only ticks counter.
- vc33 L1 is solvable in 4 clicks (lucky seed) / 82 (salience order).

## THE BIG BUG (fixed in v13, affects any old agent version)

`GameAction(6)` RAISES ValueError — arcengine GameAction enum values are
internally `(id, action_type)` TUPLES; int lookup needs
**`GameAction.from_id(int(a))`**. v1-v12's `_to_game_action` silently
failed on every int, so `_available_actions` ALWAYS fell back to "all 7
actions" — most of the budget probed phantom actions in every game,
locally AND on Kaggle. Never use `GameAction(x)` with ints.

(v8/v9 numbers in the old table were from a DIFFERENT harness in the Linux
sandbox — not comparable. v9 code is gone; v10 re-evaled here = 4 levels.)

## Diagnosis notes (v10 diag run)

- State explosion was THE bug: ls20/tr87/wa30 had 2000-3200 "states" in
  4000 actions (UI counters/energy bars make every frame unique) → momentum
  fired 53-79% of actions, no-op detection dead. v11's CellVolatility mask
  (cells changing in >=20% of frames are hashed as 0xFF) fixes this.
- Avatar detection NEVER worked in v10/v11 (avatar_color None / dirmap {}
  in every keyboard game): AvatarModel._objects required the avatar color
  to form exactly ONE component grid-wide, but borders/UI share the color.
  v12 rewrites detection: diff consecutive frames, bbox the changed region,
  match color masses inside the bbox only; ignores colors with >300 cells
  (backgrounds anti-move). locate() = component nearest last pos.
- Click games (vc33 etc.): few states (241 in vc33), most clicks noop.
  Click-color productivity tiers added in v11 (vc33 got its 1st level).

## How to run evaluation (Windows, this machine)

```bash
cd /c/Users/bader/Downloads/kaggle && python eval_harness.py --all --agent my_agent_v12.py
```

Engine facts:
- available_actions may arrive as ints — my_agent normalizes them.
- Frames are numpy arrays locally; harness converts to lists (Kaggle parity).
- Competition scoring: per-level, weighted by level index (1-based),
  (baseline/actions)^2 capped 115; completing levels FAST matters.
- FrameDataRaw fields: game_id, state, levels_completed, win_levels,
  available_actions, frame (property). No .score — harness shim sets
  score=levels_completed.

## README rule (STANDING)

`Downloads/kaggle/README.md` is the public-facing explanation of the agent
and is embedded as the FIRST cell of the Kaggle notebook
(build_notebook.py reads ~/kaggle/arc/README.md). **Every champion change
must update README.md** (version number + version-history row + any
architecture changes), then: copy README+agent to WSL, rebuild notebook,
push kernel.

## Deploy/submit flow (user runs in WSL)

```bash
cp /mnt/c/Users/bader/Downloads/kaggle/my_agent.py ~/kaggle/arc/agent/my_agent.py
cd ~/kaggle/arc && python3 scripts/build_notebook.py
KAGGLE_CONFIG_DIR=~/kaggle/arc/ARC-AGI-3-Kaggle-Starter/.kaggle \
  ~/kaggle/arc/ARC-AGI-3-Kaggle-Starter/.venv/bin/kaggle kernels push -p notebooks/
# kaggle.com -> notebook -> Submit to Competition -> submission.parquet
```

Kernel id in ~/kaggle/arc/notebooks/kernel-metadata.json must match the
existing kernel slug (409 Conflict = title-slug vs id mismatch).

User's local eval (alternative to sandbox eval):
```bash
cd ~/kaggle/arc/ARC-AGI-3-Kaggle-Starter
cp ~/kaggle/arc/agent/my_agent.py agent/my_agent.py
.venv/bin/python scripts/play_local.py --max-steps 4000 2>/dev/null | tail -28
```

## Next steps, in order

1. Evaluate v10 on the 25 cached games (sandbox or user's WSL). Compare vs
   v9 (5 levels / 0.00149). Champion = better one.
2. Iterate: the roadmap is (a) rule induction across levels of one game,
   (b) goal detection improvements, (c) mine Kaggle episode replays after
   each submission. Momentum/fixation knobs are DONE, don't re-tune them.
3. Submit the champion daily (1/day). Kaggle MCP connector exists but its
   OAuth never worked (Unauthenticated on user-scoped calls); the working
   path is kaggle CLI in the user's WSL (token in starter's .kaggle/).
4. Rules of the loop: never submit a version that didn't beat the previous
   champion locally; levels-completed count > aggregate for comparing
   (aggregate is speed-sensitive noise at current scale).

## Warnings

- Do NOT connect WSL (\\wsl.localhost\...) folders to Cowork — UNC mounts
  crash the sandbox shell for the whole session.
- The user's KGAT token has appeared in chat; remind them to rotate it at
  kaggle.com/settings when the competition workflow stabilizes.
