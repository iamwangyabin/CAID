from __future__ import annotations

import argparse
import csv
from pathlib import Path


def infer_label(path: Path) -> int:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "fake" in parts or "1_fake" in parts or "synthetic" in parts or "ai" in parts or "fake" in name:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple CAIDBench manifest from folder trees")
    parser.add_argument("--root", required=True, help="Dataset root")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.add_argument("--task-glob", default="*", help="Top-level task/domain folder glob")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split assigned to all discovered files")
    parser.add_argument("--ext", nargs="*", default=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".npy", ".pt", ".pth"])
    args = parser.parse_args()
    root = Path(args.root)
    exts = {e.lower() for e in args.ext}
    rows = []
    task_dirs = [p for p in sorted(root.glob(args.task_glob)) if p.is_dir()]
    if not task_dirs:
        task_dirs = [root]
    for task_id, task_dir in enumerate(task_dirs):
        for f in sorted(task_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                rel = f.relative_to(root)
                rows.append({
                    "path": str(rel),
                    "label": infer_label(f),
                    "split": args.split,
                    "task_id": task_id,
                    "domain": task_dir.name,
                    "generator": task_dir.name,
                    "scene": "unknown",
                })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["path", "label", "split", "task_id", "domain", "generator", "scene"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
