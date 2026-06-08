from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


Coord = tuple[int, int]
Direction = tuple[int, int]

DIRS: list[Direction] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = {(-1, 0): "up", (0, 1): "right", (1, 0): "down", (0, -1): "left", (0, 0): "stop"}

STRATEGIES = {
    "cautious": {"stop": 0.15, "best": 0.65, "side": 0.20},
    "balanced": {"stop": 0.05, "best": 0.55, "side": 0.40},
    "aggressive": {"stop": 0.02, "best": 0.78, "side": 0.20},
}


@dataclass
class Bot:
    row: int
    col: int
    direction: Direction
    ttl: int

    @property
    def pos(self) -> Coord:
        return self.row, self.col


def build_city(size: int = 30) -> list[list[int]]:
    grid = [[1 for _ in range(size)] for _ in range(size)]

    def carve_path(points: list[Coord]) -> None:
        for (r1, c1), (r2, c2) in zip(points, points[1:]):
            dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
            dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)
            r, c = r1, c1
            grid[r][c] = 0
            while (r, c) != (r2, c2):
                if r != r2:
                    r += dr
                    grid[r][c] = 0
                if c != c2:
                    c += dc
                grid[r][c] = 0

    # Outer ring road with an asymmetric inner district and diagonal avenue.
    carve_path([(3, 3), (3, 25), (8, 25), (8, 28), (18, 28), (18, 22), (26, 22), (26, 4), (20, 4), (20, 2), (9, 2), (9, 7), (3, 7), (3, 3)])
    carve_path([(5, 11), (5, 20), (11, 20), (11, 15), (16, 15), (16, 24)])
    carve_path([(23, 6), (23, 16), (18, 16), (18, 20)])
    carve_path([(6, 4), (12, 10), (17, 13), (24, 20)])
    carve_path([(11, 4), (11, 12), (14, 12)])
    carve_path([(14, 6), (14, 18), (9, 18)])
    carve_path([(21, 8), (17, 8), (17, 5)])
    carve_path([(24, 12), (27, 12), (27, 17)])
    carve_path([(6, 23), (13, 23), (13, 27)])

    # Short dead-end service lanes make bot behavior less predictable.
    carve_path([(5, 14), (2, 14)])
    carve_path([(8, 25), (5, 27)])
    carve_path([(16, 15), (14, 17)])
    carve_path([(23, 10), (26, 10)])
    carve_path([(11, 8), (8, 8)])

    starts = [(3, 3), (20, 2), (27, 17)]
    spawns = [(3, 25), (8, 28), (18, 28), (26, 4), (2, 14), (13, 27), (26, 10)]
    for r, c in starts:
        grid[r][c] = 2
    for r, c in spawns:
        grid[r][c] = 3
    grid[24][20] = 4
    return grid


def passable(grid: list[list[int]], pos: Coord) -> bool:
    r, c = pos
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] in {0, 2, 3, 4}


def add(pos: Coord, direction: Direction) -> Coord:
    return pos[0] + direction[0], pos[1] + direction[1]


def valid_dirs(grid: list[list[int]], pos: Coord) -> list[Direction]:
    return [direction for direction in DIRS if passable(grid, add(pos, direction))]


def reverse(direction: Direction) -> Direction:
    return -direction[0], -direction[1]


def turn_options(direction: Direction) -> tuple[Direction, Direction, Direction]:
    if direction == (0, 0):
        return DIRS[0], DIRS[1], DIRS[2]
    left = (-direction[1], direction[0])
    right = (direction[1], -direction[0])
    return direction, right, left


def shortest_dirs(grid: list[list[int]], pos: Coord, destination: Coord, dirs: list[Direction]) -> list[Direction]:
    def score(direction: Direction) -> int:
        next_pos = add(pos, direction)
        return abs(next_pos[0] - destination[0]) + abs(next_pos[1] - destination[1])

    best = min(score(direction) for direction in dirs)
    return [direction for direction in dirs if score(direction) == best]


def agent_worker(conn) -> None:
    rng = random.Random()
    while True:
        msg = conn.recv()
        if msg["type"] == "stop":
            return
        grid = msg["grid"]
        pos = tuple(msg["pos"])
        destination = tuple(msg["destination"])
        prev_dir = tuple(msg["prev_dir"])
        strategy = STRATEGIES[msg["strategy"]]
        dirs = valid_dirs(grid, pos)
        if not dirs:
            conn.send((0, 0))
            continue

        roll = rng.random()
        if roll < strategy["stop"]:
            conn.send((0, 0))
            continue

        best_dirs = shortest_dirs(grid, pos, destination, dirs)
        if roll < strategy["stop"] + strategy["best"]:
            conn.send(rng.choice(best_dirs))
            continue

        straight, right, left = turn_options(prev_dir)
        side_dirs = [direction for direction in [right, left, straight] if direction in dirs]
        conn.send(rng.choice(side_dirs or dirs))


def choose_bot_direction(grid: list[list[int]], bot: Bot, rng: random.Random) -> Direction:
    dirs = valid_dirs(grid, bot.pos)
    if not dirs:
        return (0, 0)
    backward = reverse(bot.direction)
    forward_options = [direction for direction in dirs if direction != backward]
    if len(dirs) == 1:
        return dirs[0]
    if not forward_options:
        return backward
    if len(forward_options) == 1:
        return forward_options[0]
    if rng.random() < 0.10:
        return (0, 0)
    return rng.choice(forward_options)


def bot_worker(conn) -> None:
    rng = random.Random()
    while True:
        msg = conn.recv()
        if msg["type"] == "stop":
            return
        grid = msg["grid"]
        bots = [Bot(item["row"], item["col"], tuple(item["direction"]), item["ttl"]) for item in msg["bots"]]
        spawn_rate = msg["spawn_rate"]
        spawns = [tuple(item) for item in msg["spawns"]]

        updated: list[Bot] = []
        for bot in bots:
            bot.ttl -= 1
            if bot.ttl <= 0:
                continue
            direction = choose_bot_direction(grid, bot, rng)
            if direction != (0, 0):
                bot.row, bot.col = add(bot.pos, direction)
                bot.direction = direction
            updated.append(bot)

        for spawn in spawns:
            if rng.random() >= spawn_rate:
                continue
            dirs = valid_dirs(grid, spawn)
            if not dirs:
                continue
            direction = rng.choice(dirs)
            updated.append(Bot(spawn[0], spawn[1], direction, rng.randint(15, 150)))

        conn.send([
            {"row": bot.row, "col": bot.col, "direction": bot.direction, "ttl": bot.ttl}
            for bot in updated
        ])


def find_cells(grid: list[list[int]], value: int) -> list[Coord]:
    return [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == value]


def render_frame(grid: list[list[int]], agent: Coord, bots: list[Bot], scale: int = 16) -> Image.Image:
    colors = {0: (235, 235, 235), 1: (30, 30, 30), 2: (160, 210, 255), 3: (255, 230, 140), 4: (140, 230, 160)}
    image = Image.new("RGB", (len(grid[0]) * scale, len(grid) * scale), "white")
    draw = ImageDraw.Draw(image)
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            x0, y0 = c * scale, r * scale
            draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1], fill=colors[cell], outline=(200, 200, 200))
    for bot in bots:
        x, y = bot.col * scale, bot.row * scale
        draw.ellipse([x + 3, y + 3, x + scale - 4, y + scale - 4], fill=(220, 80, 60))
    x, y = agent[1] * scale, agent[0] * scale
    draw.rectangle([x + 3, y + 3, x + scale - 4, y + scale - 4], fill=(60, 110, 230))
    return image


def run_episode(
    strategy: str,
    spawn_rate: float,
    seed: int,
    max_steps: int = 700,
    capture: bool = False,
) -> dict:
    grid = build_city()
    starts = find_cells(grid, 2)
    spawns = find_cells(grid, 3)
    destination = find_cells(grid, 4)[0]
    rng = random.Random(seed)
    agent = rng.choice(starts)
    agent_dir: Direction = (0, 0)
    bots: list[Bot] = []
    frames: list[Image.Image] = []

    agent_parent, agent_child = mp.Pipe()
    bot_parent, bot_child = mp.Pipe()
    agent_proc = mp.Process(target=agent_worker, args=(agent_child,))
    bot_proc = mp.Process(target=bot_worker, args=(bot_child,))
    agent_proc.start()
    bot_proc.start()

    success = False
    crash = False
    try:
        for step in range(1, max_steps + 1):
            if capture and step % 4 == 1:
                frames.append(render_frame(grid, agent, bots))

            agent_parent.send({
                "type": "step",
                "grid": grid,
                "pos": agent,
                "destination": destination,
                "prev_dir": agent_dir,
                "strategy": strategy,
            })
            bot_parent.send({
                "type": "step",
                "grid": grid,
                "bots": [{"row": b.row, "col": b.col, "direction": b.direction, "ttl": b.ttl} for b in bots],
                "spawn_rate": spawn_rate,
                "spawns": spawns,
            })
            move = tuple(agent_parent.recv())
            bot_payload = bot_parent.recv()
            new_agent = add(agent, move) if move != (0, 0) and passable(grid, add(agent, move)) else agent
            if move != (0, 0) and new_agent != agent:
                agent_dir = move
            bots = [Bot(item["row"], item["col"], tuple(item["direction"]), item["ttl"]) for item in bot_payload]

            positions: dict[Coord, int] = {}
            filtered_bots: list[Bot] = []
            collided_positions: set[Coord] = set()
            for bot in bots:
                if bot.pos in positions:
                    collided_positions.add(bot.pos)
                else:
                    positions[bot.pos] = 1
                    filtered_bots.append(bot)
            bots = [bot for bot in filtered_bots if bot.pos not in collided_positions]

            agent = new_agent
            if agent in {bot.pos for bot in bots} or agent in collided_positions:
                crash = True
                break
            if agent == destination:
                success = True
                break
        else:
            step = max_steps
    finally:
        agent_parent.send({"type": "stop"})
        bot_parent.send({"type": "stop"})
        agent_proc.join(timeout=2)
        bot_proc.join(timeout=2)
        if agent_proc.is_alive():
            agent_proc.terminate()
        if bot_proc.is_alive():
            bot_proc.terminate()

    return {
        "strategy": strategy,
        "spawn_rate": spawn_rate,
        "seed": seed,
        "success": success,
        "crash": crash,
        "steps": step,
        "frames": frames,
    }


def draw_chart(rows: list[dict], out_path: Path) -> None:
    strategies = list(STRATEGIES)
    rates = sorted({row["spawn_rate"] for row in rows})
    width, height = 900, 560
    margin = 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = margin, 50, width - 40, height - margin
    draw.rectangle([left, top, right, bottom], outline="black")
    max_steps = max(row["avg_steps_success"] for row in rows if row["avg_steps_success"] is not None) * 1.2

    def px(rate: float) -> float:
        return left + (rate - min(rates)) / (max(rates) - min(rates)) * (right - left)

    def py(value: float) -> float:
        return bottom - value / max_steps * (bottom - top)

    colors = {"cautious": (30, 90, 180), "balanced": (30, 150, 80), "aggressive": (210, 90, 40)}
    for i in range(6):
        y = max_steps * i / 5
        yy = py(y)
        draw.line([left, yy, right, yy], fill=(225, 225, 225))
        draw.text((10, yy - 7), f"{y:.0f}", fill="black")
    for strategy in strategies:
        points = []
        for rate in rates:
            row = next(item for item in rows if item["strategy"] == strategy and item["spawn_rate"] == rate)
            value = row["avg_steps_success"] or row["avg_steps_all"]
            points.append((px(rate), py(value)))
        draw.line(points, fill=colors[strategy], width=3)
        for x, y in points:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=colors[strategy])
        draw.text((right - 150, top + 20 + 18 * strategies.index(strategy)), strategy, fill=colors[strategy])
    for rate in rates:
        draw.text((px(rate) - 15, bottom + 12), f"{rate:.2f}", fill="black")
    draw.text((width // 2 - 160, 15), "Average successful delivery steps", fill="black")
    draw.text((width // 2 - 80, height - 35), "Bot spawn probability", fill="black")
    draw.text((8, 18), "Steps", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def run_experiments(episodes: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = [0.01, 0.03, 0.06]
    rows: list[dict] = []
    details: list[dict] = []
    for strategy in STRATEGIES:
        for rate in rates:
            results = [run_episode(strategy, rate, seed=1000 + i + int(rate * 1000), capture=False) for i in range(episodes)]
            successes = [item for item in results if item["success"]]
            row = {
                "strategy": strategy,
                "spawn_rate": rate,
                "episodes": episodes,
                "success_rate": len(successes) / episodes,
                "avg_steps_success": round(statistics.mean([item["steps"] for item in successes]), 2) if successes else None,
                "avg_steps_all": round(statistics.mean([item["steps"] for item in results]), 2),
            }
            rows.append(row)
            details.extend([{k: v for k, v in item.items() if k != "frames"} for item in results])
            print(row)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_chart(rows, out_dir / "avg_steps.png")

    demo = run_episode("balanced", 0.03, seed=777, capture=True)
    if demo["frames"]:
        demo["frames"][0].save(
            out_dir / "movement_demo.gif",
            save_all=True,
            append_images=demo["frames"][1:],
            duration=90,
            loop=0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", default="lab4_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(args.episodes, Path(args.out))


if __name__ == "__main__":
    mp.freeze_support()
    main()
