from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import ClientSession, ClientTimeout, web
from PIL import Image, ImageDraw, ImageFont


STRATEGIES = {"fixed", "timeout_race", "adaptive"}


@dataclass
class EndpointProfile:
    name: str
    min_delay_ms: int
    max_delay_ms: int
    error_rate: float = 0.0
    status_code: int = 503


class SlidingStats:
    def __init__(self, window: int = 20, initial_concurrency: int = 3) -> None:
        self.window = window
        self.samples: deque[tuple[bool, float]] = deque(maxlen=window)
        self.adjusted_concurrency = initial_concurrency

    def record(self, success: bool, elapsed_ms: float) -> None:
        self.samples.append((success, elapsed_ms))

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def success_rate(self) -> float:
        if not self.samples:
            return 1.0
        return sum(1 for success, _ in self.samples if success) / len(self.samples)

    @property
    def avg_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(elapsed for _, elapsed in self.samples) / len(self.samples)

    def adapt(self, global_limit: int) -> int:
        if self.count < 3:
            self.adjusted_concurrency = min(global_limit, max(1, self.adjusted_concurrency))
            return self.adjusted_concurrency

        if self.success_rate < 0.70 or self.avg_ms > 2000:
            self.adjusted_concurrency = max(1, self.adjusted_concurrency - 1)
        elif self.success_rate > 0.90 and self.avg_ms < 500:
            self.adjusted_concurrency = min(global_limit, self.adjusted_concurrency + 1)
        else:
            self.adjusted_concurrency = min(global_limit, max(1, self.adjusted_concurrency))
        return self.adjusted_concurrency

    def snapshot(self) -> dict[str, Any]:
        return {
            "success_rate": round(self.success_rate, 3),
            "avg_ms": round(self.avg_ms, 2),
            "samples": self.count,
            "adjusted_concurrency": self.adjusted_concurrency,
        }


class AdaptiveStatsStore:
    def __init__(self, window: int = 20, initial_concurrency: int = 3) -> None:
        self.window = window
        self.initial_concurrency = initial_concurrency
        self._stats: dict[str, SlidingStats] = {}

    def get(self, key: str) -> SlidingStats:
        if key not in self._stats:
            self._stats[key] = SlidingStats(self.window, self.initial_concurrency)
        return self._stats[key]

    def record(self, key: str, success: bool, elapsed_ms: float, global_limit: int) -> None:
        stat = self.get(key)
        stat.record(success, elapsed_ms)
        stat.adapt(global_limit)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {key: stat.snapshot() for key, stat in sorted(self._stats.items())}

    def reset(self) -> None:
        self._stats.clear()


def api_key(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


async def fetch_url(session: ClientSession, url: str, timeout_sec: float) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        async with session.get(url, timeout=ClientTimeout(total=timeout_sec)) as response:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if response.content_type == "application/json":
                data = await response.json()
            else:
                data = await response.text()
            result: dict[str, Any] = {"url": url, "status": response.status, "elapsed_ms": round(elapsed_ms, 2)}
            if 200 <= response.status < 300:
                result["data"] = data
            else:
                result["error"] = data
            return result
    except asyncio.TimeoutError:
        return {"url": url, "timeout": True, "elapsed_ms": round(timeout_sec * 1000, 2)}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"url": url, "error": str(exc), "elapsed_ms": round(elapsed_ms, 2)}


def is_success(result: dict[str, Any]) -> bool:
    return result.get("status", 0) >= 200 and result.get("status", 0) < 300 and not result.get("timeout")


async def run_fixed(urls: list[str], max_concurrent: int, timeout_sec: float, store: AdaptiveStatsStore | None = None) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    async with ClientSession() as session:
        async def guarded(url: str) -> dict[str, Any]:
            async with semaphore:
                return await fetch_url(session, url, timeout_sec)

        return await asyncio.gather(*(guarded(url) for url in urls))


async def run_timeout_race(urls: list[str], timeout_sec: float) -> list[dict[str, Any]]:
    async with ClientSession() as session:
        tasks = [asyncio.create_task(fetch_url(session, url, timeout_sec)) for url in urls]
        return await asyncio.gather(*tasks)


async def run_adaptive(urls: list[str], max_concurrent: int, timeout_sec: float, store: AdaptiveStatsStore) -> list[dict[str, Any]]:
    by_key: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        by_key[api_key(url)].append(url)

    semaphores = {
        key: asyncio.Semaphore(store.get(key).adapt(max_concurrent))
        for key in by_key
    }

    async with ClientSession() as session:
        async def guarded(url: str) -> dict[str, Any]:
            key = api_key(url)
            async with semaphores[key]:
                result = await fetch_url(session, url, timeout_sec)
                store.record(key, is_success(result), result["elapsed_ms"], max_concurrent)
                return result

        return await asyncio.gather(*(guarded(url) for url in urls))


async def aggregate(
    urls: list[str],
    strategy: str,
    max_concurrent: int,
    timeout_sec: float,
    store: AdaptiveStatsStore,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    start = time.perf_counter()
    if strategy == "fixed":
        results = await run_fixed(urls, max_concurrent, timeout_sec)
    elif strategy == "timeout_race":
        results = await run_timeout_race(urls, timeout_sec)
    else:
        results = await run_adaptive(urls, max_concurrent, timeout_sec, store)
    total_time_ms = (time.perf_counter() - start) * 1000

    successful = sum(1 for item in results if is_success(item))
    failed = len(results) - successful
    return {
        "request_id": str(uuid.uuid4()),
        "results": results,
        "summary": {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "timeouts": sum(1 for item in results if item.get("timeout")),
            "total_time_ms": round(total_time_ms, 2),
            "strategy_used": strategy,
            "concurrent_used": max_concurrent,
        },
        "adaptive_stats": store.snapshot(),
    }


def create_gateway_app(store: AdaptiveStatsStore | None = None) -> web.Application:
    app = web.Application()
    app["stats_store"] = store or AdaptiveStatsStore()

    async def aggregate_handler(request: web.Request) -> web.Response:
        payload = await request.json()
        urls = payload.get("urls") or []
        if not isinstance(urls, list) or not urls:
            raise web.HTTPBadRequest(text="urls must be a non-empty list")
        strategy = payload.get("strategy", "fixed")
        max_concurrent = int(payload.get("max_concurrent", 3))
        timeout_sec = float(payload.get("timeout_sec", 5))
        data = await aggregate(urls, strategy, max_concurrent, timeout_sec, request.app["stats_store"])
        return web.json_response(data)

    async def stats_handler(request: web.Request) -> web.Response:
        return web.json_response(request.app["stats_store"].snapshot())

    app.router.add_post("/aggregate", aggregate_handler)
    app.router.add_get("/stats", stats_handler)
    return app


def create_emulator_app(profile: EndpointProfile) -> web.Application:
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        delay = random.uniform(profile.min_delay_ms, profile.max_delay_ms) / 1000
        await asyncio.sleep(delay)
        if random.random() < profile.error_rate:
            return web.json_response({"service": profile.name, "error": "simulated failure"}, status=profile.status_code)
        return web.json_response({"service": profile.name, "value": random.randint(1, 100), "delay_ms": round(delay * 1000, 2)})

    app.router.add_get("/data", handler)
    return app


class EmulatorCluster:
    def __init__(self, profiles: list[EndpointProfile]) -> None:
        self.profiles = profiles
        self.runners: list[web.AppRunner] = []
        self.sites: list[web.TCPSite] = []
        self.urls: list[str] = []

    async def __aenter__(self) -> "EmulatorCluster":
        for profile in self.profiles:
            runner = web.AppRunner(create_emulator_app(profile))
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            sockets = site._server.sockets
            port = sockets[0].getsockname()[1]
            self.runners.append(runner)
            self.sites.append(site)
            self.urls.append(f"http://127.0.0.1:{port}/data")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        for runner in self.runners:
            await runner.cleanup()


SCENARIOS = {
    "stable": [
        EndpointProfile("stable-a", 50, 140, 0.0),
        EndpointProfile("stable-b", 70, 160, 0.0),
        EndpointProfile("stable-c", 80, 180, 0.0),
    ],
    "slow": [
        EndpointProfile("fast-a", 70, 160, 0.0),
        EndpointProfile("slow-b", 3000, 5000, 0.0),
        EndpointProfile("fast-c", 80, 180, 0.0),
    ],
    "unstable": [
        EndpointProfile("stable-a", 60, 170, 0.0),
        EndpointProfile("flaky-b", 100, 350, 0.40, 503),
        EndpointProfile("flaky-c", 100, 400, 0.35, 500),
    ],
}


async def run_scenario(name: str, repeats: int) -> dict[str, Any]:
    strategy_rows: list[dict[str, Any]] = []
    adaptive_trace: list[dict[str, Any]] = []
    profiles = SCENARIOS[name]

    async with EmulatorCluster(profiles) as cluster:
        urls = cluster.urls
        for strategy in ["fixed", "timeout_race", "adaptive"]:
            store = AdaptiveStatsStore(initial_concurrency=3)
            for repeat in range(1, repeats + 1):
                response = await aggregate(urls, strategy, 3, 2.0, store)
                summary = response["summary"]
                row = {
                    "scenario": name,
                    "strategy": strategy,
                    "repeat": repeat,
                    "total_time_ms": summary["total_time_ms"],
                    "successful": summary["successful"],
                    "failed": summary["failed"],
                    "timeouts": summary["timeouts"],
                }
                strategy_rows.append(row)
                if strategy == "adaptive":
                    for key, stat in response["adaptive_stats"].items():
                        adaptive_trace.append({
                            "scenario": name,
                            "repeat": repeat,
                            "api": key,
                            "concurrency": stat["adjusted_concurrency"],
                            "success_rate": stat["success_rate"],
                            "avg_ms": stat["avg_ms"],
                        })

    return {"rows": strategy_rows, "adaptive_trace": adaptive_trace}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["strategy"])].append(row)

    result = []
    for (scenario, strategy), items in sorted(grouped.items()):
        result.append({
            "scenario": scenario,
            "strategy": strategy,
            "avg_total_time_ms": round(statistics.mean(item["total_time_ms"] for item in items), 2),
            "avg_successful": round(statistics.mean(item["successful"] for item in items), 2),
            "avg_failed": round(statistics.mean(item["failed"] for item in items), 2),
            "avg_timeouts": round(statistics.mean(item["timeouts"] for item in items), 2),
        })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_bar_chart(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    width, height = 1100, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin = 80
    left, top, right, bottom = margin, 50, width - 40, height - 110
    max_value = max(row["avg_total_time_ms"] for row in summary_rows) * 1.15
    draw.rectangle([left, top, right, bottom], outline="black")
    colors = {"fixed": (50, 100, 200), "timeout_race": (220, 130, 40), "adaptive": (60, 160, 90)}
    scenarios = list(SCENARIOS)
    group_width = (right - left) / len(scenarios)
    bar_width = group_width / 5
    for i in range(6):
        value = max_value * i / 5
        y = bottom - value / max_value * (bottom - top)
        draw.line([left, y, right, y], fill=(225, 225, 225))
        draw.text((10, y - 7), f"{value:.0f}", fill="black", font=font)
    for si, scenario in enumerate(scenarios):
        rows = [row for row in summary_rows if row["scenario"] == scenario]
        for bi, row in enumerate(rows):
            x0 = left + si * group_width + 45 + bi * bar_width
            y0 = bottom - row["avg_total_time_ms"] / max_value * (bottom - top)
            draw.rectangle([x0, y0, x0 + bar_width - 6, bottom], fill=colors[row["strategy"]])
            draw.text((x0 - 4, y0 - 16), f"{row['avg_total_time_ms']:.0f}", fill="black", font=font)
        draw.text((left + si * group_width + 50, bottom + 20), scenario, fill="black", font=font)
    for i, (strategy, color) in enumerate(colors.items()):
        draw.rectangle([right - 170, top + 20 + i * 18, right - 158, top + 32 + i * 18], fill=color)
        draw.text((right - 150, top + 18 + i * 18), strategy, fill="black", font=font)
    draw.text((width // 2 - 135, 20), "Average aggregation time by strategy", fill="black", font=font)
    draw.text((12, 18), "Time, ms", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_adaptive_chart(path: Path, trace_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in trace_rows if row["scenario"] == "unstable"]
    if not rows:
        return
    width, height = 1100, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = 80, 50, width - 40, height - 70
    draw.rectangle([left, top, right, bottom], outline="black")
    apis = sorted({row["api"] for row in rows})
    colors = [(40, 100, 200), (220, 100, 50), (40, 160, 80)]
    max_repeat = max(row["repeat"] for row in rows)
    max_concurrency = max(row["concurrency"] for row in rows) + 1

    def px(repeat: int) -> float:
        return left + (repeat - 1) / max(1, max_repeat - 1) * (right - left)

    def py(value: float) -> float:
        return bottom - value / max_concurrency * (bottom - top)

    for i in range(max_concurrency + 1):
        y = py(i)
        draw.line([left, y, right, y], fill=(225, 225, 225))
        draw.text((25, y - 7), str(i), fill="black", font=font)
    for api, color in zip(apis, colors):
        api_rows = sorted([row for row in rows if row["api"] == api], key=lambda row: row["repeat"])
        points = [(px(row["repeat"]), py(row["concurrency"])) for row in api_rows]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
        draw.text((right - 360, top + 18 + 18 * apis.index(api)), api[-32:], fill=color, font=font)
    draw.text((width // 2 - 155, 20), "Adaptive concurrency in unstable scenario", fill="black", font=font)
    draw.text((width // 2 - 40, height - 35), "Repeat", fill="black", font=font)
    draw.text((10, 18), "Concurrency", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


async def run_tests_cli(args: argparse.Namespace) -> None:
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    all_rows: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    for scenario in scenarios:
        data = await run_scenario(scenario, args.repeats)
        all_rows.extend(data["rows"])
        all_trace.extend(data["adaptive_trace"])

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(all_rows)
    write_csv(out / "raw_results.csv", all_rows)
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "adaptive_trace.csv", all_trace)
    (out / "results.json").write_text(json.dumps({"rows": all_rows, "summary": summary_rows, "adaptive_trace": all_trace}, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_bar_chart(out / "avg_time_by_strategy.png", summary_rows)
    draw_adaptive_chart(out / "adaptive_concurrency.png", all_trace)
    print(json.dumps({"summary": summary_rows}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    test = sub.add_parser("test")
    test.add_argument("--scenario", choices=["all", *SCENARIOS.keys()], default="all")
    test.add_argument("--repeats", type=int, default=10)
    test.add_argument("--output", default="lab5_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "serve":
        web.run_app(create_gateway_app(), host=args.host, port=args.port)
    elif args.command == "test":
        asyncio.run(run_tests_cli(args))


if __name__ == "__main__":
    main()
