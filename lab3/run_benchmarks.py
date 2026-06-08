from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


COUNTS = [2, 3, 6]
OUT_DIR = Path("lab3_results")


def mpi_command(processes: int, out_path: Path) -> list[str]:
    python_dir = Path(sys.executable).resolve().parent
    local_mpiexec = python_dir / ("mpiexec.exe" if os_name_is_windows() else "mpiexec")
    local_mpirun = python_dir / ("mpirun.exe" if os_name_is_windows() else "mpirun")
    launcher = (
        str(local_mpiexec) if local_mpiexec.exists()
        else str(local_mpirun) if local_mpirun.exists()
        else shutil.which("mpiexec") or shutil.which("mpirun")
    )
    if not launcher:
        raise RuntimeError("mpiexec/mpirun was not found in PATH")
    return [
        launcher,
        "-n",
        str(processes),
        sys.executable,
        "periodic_mpi.py",
        "--out",
        str(out_path),
    ]


def os_name_is_windows() -> bool:
    return sys.platform.startswith("win")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for processes in COUNTS:
        out_path = OUT_DIR / f"run_{processes}.json"
        cmd = mpi_command(processes, out_path)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "processes": processes,
                "elapsed_s": data["elapsed_s"],
                "elements_processed": data["elements_processed"],
            }
        )

    summary = {"runs": rows}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
