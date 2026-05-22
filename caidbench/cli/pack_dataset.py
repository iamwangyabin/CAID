from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.arrow_schema import CANONICAL_COLUMNS, write_arrow_table
from ..data.dataset_packers import PACKERS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw/processed datasets into AID-style Arrow/HF datasets plus JSON sidecars."
    )
    parser.add_argument("--kind", required=True, choices=sorted(PACKERS), help="Dataset packer to use")
    parser.add_argument("--root", default=None, help="Dataset root. Required for scanner-based packers and path resolution")
    parser.add_argument("--manifest", default=None, help="Input CSV/JSONL manifest when --kind manifest")
    parser.add_argument("--out", required=True, help="Output AID-style HF dataset directory")
    parser.add_argument("--format", choices=["aid", "arrow", "parquet"], default="aid")
    parser.add_argument("--dataset-name", default="manifest", help="Dataset name for manifest packer")
    parser.add_argument("--embed-images", action="store_true", help="Deprecated/no-op: AID-style output always stores image bytes in column `image`")
    parser.add_argument("--compute-sha1", action="store_true", help="Compute image SHA1 hashes")
    parser.add_argument("--no-size", action="store_true", help="Do not open images to record width/height")
    parser.add_argument("--strict-images", action="store_true", help="Fail if referenced images are missing")
    parser.add_argument("--default-split", default="train", help="Split assigned when no split folder is found")
    parser.add_argument("--preprocess-profile", default="", help="Value written to preprocess_profile column")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug: limit number of scanned samples")
    parser.add_argument("--dry-run", action="store_true", help="Only print schema/counts; do not write Arrow")
    args = parser.parse_args()

    packer = PACKERS[args.kind]
    common = dict(
        embed_images=args.embed_images,
        compute_sha1=args.compute_sha1,
        compute_size=not args.no_size,
        strict_images=args.strict_images,
        default_split=args.default_split,
        preprocess_profile=args.preprocess_profile,
        max_samples=args.max_samples,
    )
    if args.kind == "manifest":
        if not args.manifest:
            raise SystemExit("--kind manifest requires --manifest")
        df = packer(args.manifest, root=args.root, dataset_name=args.dataset_name, **common)
    else:
        if not args.root:
            raise SystemExit(f"--kind {args.kind} requires --root")
        df = packer(args.root, **common)

    # Data quality report.
    bad_label = int((df["label"].astype(int) < 0).sum())
    report = {
        "rows": int(len(df)),
        "arrow_columns": ["image"] if args.format == "aid" else list(df.columns),
        "sidecar_index_columns": CANONICAL_COLUMNS,
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).sort_index().items()},
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts(dropna=False).items()},
        "dataset_counts": {str(k): int(v) for k, v in df["dataset"].value_counts(dropna=False).head(20).items()},
        "generator_counts": {str(k): int(v) for k, v in df["generator"].value_counts(dropna=False).head(30).items()},
        "unknown_label_rows": bad_label,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if bad_label:
        print("WARNING: some rows have label=-1. Fix folders/manifest or filter them before training.")
    if args.dry_run:
        return
    out = write_arrow_table(df, args.out, fmt=args.format, root=args.root)
    print(f"Wrote {len(df)} normalized rows to {out}")


if __name__ == "__main__":
    main()
