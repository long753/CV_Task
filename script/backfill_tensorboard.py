#!/home/long753/miniconda3/envs/cv_task/bin/python
"""Import a stopped run's completed CSV epochs before enabling live TensorBoard logging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Establish project-local Ultralytics/Matplotlib settings before optional imports.
from script.train_yolo import configure_ultralytics
from torch.utils.tensorboard import SummaryWriter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Stopped training run directory")
    return parser.parse_args(argv)


def load_completed_scalars(run_dir: Path) -> list[tuple[int, dict[str, float]]]:
    """Read genuine completed epochs, keeping the native scalar names and one-based steps."""
    state = json.loads((run_dir / "training_state.json").read_text(encoding="utf-8"))
    completed = int(state["completed_epochs"])
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tags = [name for name in reader.fieldnames or [] if name not in {"epoch", "time"}]
        records = []
        for row in reader:
            step = int(row["epoch"])
            if step > completed:
                continue
            scalars = {tag: float(row[tag]) for tag in tags}
            if not all(math.isfinite(value) for value in scalars.values()):
                raise ValueError(f"Non-finite CSV metric at epoch {step}")
            records.append((step, scalars))
    if not records or not tags or [step for step, _ in records] != list(range(1, completed + 1)):
        raise ValueError("CSV must contain every completed epoch exactly once in order")
    return records


def backfill_tensorboard(run_dir: Path) -> dict[str, object]:
    """Write history once into the same directory that the native live callback uses."""
    run_dir = run_dir.resolve()
    if any(run_dir.glob("events.out.tfevents.*")):
        raise FileExistsError("Event files already exist; refusing to duplicate history or race live logging")
    records = load_completed_scalars(run_dir)
    with SummaryWriter(log_dir=str(run_dir)) as writer:
        for step, scalars in records:
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, global_step=step)
        writer.flush()
    return {
        "run_dir": str(run_dir),
        "epochs": len(records),
        "first_step": records[0][0],
        "last_step": records[-1][0],
        "tags": list(records[0][1]),
        "wall_time_note": "History is imported now; use the Step axis, not import wall time, for epoch curves.",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(backfill_tensorboard(args.run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
