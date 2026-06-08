# Lab 3: Distributed MPI Web Parsing

Variant 5 processes Wikipedia pages for chemical elements whose symbols contain two letters.

The root MPI process downloads the periodic table page, extracts element links, filters elements by two-letter symbols, and distributes element pages among MPI ranks. Each rank downloads and parses its assigned pages, then the root rank aggregates:

- top 5 elements whose own symbol is mentioned most often on their own page;
- top 5 symbols mentioned most often across all processed element pages.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
mpiexec -n 2 python periodic_mpi.py --out lab3_results/run_2.json
mpiexec -n 3 python periodic_mpi.py --out lab3_results/run_3.json
mpiexec -n 6 python periodic_mpi.py --out lab3_results/run_6.json
```

A helper script runs all required process counts:

```bash
python run_benchmarks.py
```

Generated caches and results are ignored by git.
