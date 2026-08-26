"""Render an ARC-AGI-3 frame as compact text for an LLM."""
from __future__ import annotations
from collections import defaultdict

# ARC color letters (0-15). Background is usually the most common.
LETTERS = "0123456789ABCDEF"


def to_letter_grid(grid: list[list[int]]) -> str:
    return "\n".join("".join(LETTERS[v & 0xF] for v in row) for row in grid)


def color_counts(grid):
    c = defaultdict(int)
    for row in grid:
        for v in row:
            c[v] += 1
    return c


def segment(grid: list[list[int]], max_objects: int = 40) -> list[dict]:
    """4-connected same-color components (excluding background)."""
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    counts = color_counts(grid)
    bg = max(counts, key=counts.get)
    seen = [[False] * w for _ in range(h)]
    objs = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or grid[y0][x0] == bg:
                continue
            color = grid[y0][x0]
            stack = [(x0, y0)]
            seen[y0][x0] = True
            xs = ys = 0
            n = 0
            minx = maxx = x0
            miny = maxy = y0
            while stack:
                x, y = stack.pop()
                xs += x; ys += y; n += 1
                minx = min(minx, x); maxx = max(maxx, x)
                miny = min(miny, y); maxy = max(maxy, y)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] \
                            and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            objs.append({
                "color": LETTERS[color & 0xF],
                "n": n,
                "center": [ys // n, xs // n],   # row, col
                "bbox": [miny, minx, maxy, maxx],
                "shape": [maxy - miny + 1, maxx - minx + 1],
            })
    objs.sort(key=lambda o: (o["center"][0], o["center"][1]))
    return objs[:max_objects]


def render_observation(grid, valid_actions, history, last_change,
                       max_ascii=64):
    """Compact, decision-oriented text block for the LLM."""
    if not grid:
        return "EMPTY FRAME"
    counts = color_counts(grid)
    bg = max(counts, key=counts.get)
    objs = segment(grid)
    lines = []
    lines.append(f"BOARD {len(grid)}x{len(grid[0])} background=color {bg} ({LETTERS[bg & 0xF]})")
    lines.append(f"VALID ACTIONS: {', '.join(valid_actions)}")
    if last_change is not None:
        lines.append(f"LAST ACTION changed {last_change} cells" if last_change
                     else "LAST ACTION changed nothing visible")
    lines.append(f"OBJECTS ({len(objs)} non-background components, row,col centers):")
    for i, o in enumerate(objs):
        lines.append(f"  #{i} color {o['color']} size {o['n']} center ({o['center'][0]},{o['center'][1]}) shape {o['shape'][0]}x{o['shape'][1]}")
    if history:
        recent = " -> ".join(history[-8:])
        lines.append(f"RECENT ACTIONS: {recent}")
    # small ascii only if the board is not huge
    if len(grid) <= max_ascii:
        lines.append("ASCII (letters = colors):")
        lines.append(to_letter_grid(grid))
    return "\n".join(lines)
