from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..data.arrow_schema import normalize_records, write_arrow_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a manifest into AID-style Arrow/HF dataset plus sidecars")
    parser.add_argument("--manifest", required=True, help="CSV/JSONL manifest with at least path,label,split columns")
    parser.add_argument("--root", default=None, help="Root used to resolve relative image paths")
    parser.add_argument("--out", required=True, help="Output AID-style HF dataset directory")
    parser.add_argument("--embed-images", action="store_true", help="Deprecated/no-op: AID-style output always embeds image bytes as column `image`")
    parser.add_argument("--compute-sha1", action="store_true", help="Deprecated/no-op")
    parser.add_argument("--format", choices=["aid", "arrow", "parquet"], default="aid")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if manifest.suffix.lower() in {".jsonl", ".json"}:
        df0 = pd.read_json(manifest, lines=True)
    else:
        df0 = pd.read_csv(manifest)
    df = normalize_records(df0.to_dict(orient="records"), root=args.root)
    out = write_arrow_table(df, args.out, fmt=args.format, root=args.root)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
