# Lab 5: Async Aggregation Gateway

The lab implements an asynchronous HTTP aggregation gateway with dynamic balancing.

Features:

- `POST /aggregate` endpoint implemented with `aiohttp`;
- strategies: `fixed`, `timeout_race`, `adaptive`;
- per-host sliding-window statistics;
- local aiohttp API emulators for stable, slow and unstable scenarios;
- CLI benchmark runner that writes JSON/CSV and PNG charts.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run gateway:

```bash
python main.py serve --host 127.0.0.1 --port 8080
```

Run built-in tests:

```bash
python main.py test --scenario all --repeats 10 --output lab5_results
```

Generated benchmark outputs are ignored by git.
