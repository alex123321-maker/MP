# Lab 4: Multi-agent Environment with Multiprocessing

The lab implements a city traffic simulation with:

- one delivery agent controlled by a separate process;
- traffic bots controlled by a separate process;
- a coordinator process that updates the map, detects collisions and records metrics.

Run experiments:

```bash
python run_experiments.py
```

Generated results are written to `lab4_results/` and ignored by git.
