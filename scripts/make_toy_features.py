from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy")
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for task_id in range(args.tasks):
        for split, n in [("train", args.n_train), ("test", args.n_test)]:
            for i in range(n):
                y = i % 2
                mean = task_id * 0.8 + (2.0 if y else -2.0)
                x = rng.normal(loc=mean, scale=1.0, size=(args.dim,)).astype("float32")
                path = root / f"task{task_id}_{split}_{i:04d}.npy"
                np.save(path, x)
                rows.append({
                    "path": path.name,
                    "label": y,
                    "split": split,
                    "task_id": task_id,
                    "domain": f"domain{task_id}",
                    "generator": f"gen{task_id}",
                    "scene": f"scene{task_id % 2}",
                })
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["path", "label", "split", "task_id", "domain", "generator", "scene"])
        writer.writeheader()
        writer.writerows(rows)
    print(manifest)


if __name__ == "__main__":
    main()
