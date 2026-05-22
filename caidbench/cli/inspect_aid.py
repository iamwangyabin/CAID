from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.arrow_schema import read_aid_split_sidecars


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an AID-compatible Arrow dataset directory")
    parser.add_argument("--root", "--path", dest="root", required=True, help="AID Arrow dataset directory")
    parser.add_argument("--max-subsets", type=int, default=200, help="Maximum number of subset rows to print")
    args = parser.parse_args()

    root = Path(args.root)
    df = read_aid_split_sidecars(root)
    rows = []
    for split, sdf in df.groupby("split"):
        counts = {}
        for subsets in sdf.get("subset", []):
            for sub in str(subsets).split(";"):
                if sub:
                    counts[sub] = counts.get(sub, 0) + 1
        for subset, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.max_subsets]:
            subdf = sdf[sdf["subset"].astype(str).map(lambda x, s=subset: s in set(x.split(";")))]
            label_counts = {str(k): int(v) for k, v in subdf["label"].value_counts(dropna=False).sort_index().items()}
            rows.append({"split": str(split), "subset": subset, "num_samples": int(n), "label_counts": label_counts})
    report = {
        "root": str(root),
        "num_rows_in_split_metadata": int(len(df)),
        "num_unique_arrow_rows": int(df["_rowid"].nunique()) if "_rowid" in df.columns else None,
        "splits": sorted(str(x) for x in df["split"].dropna().unique().tolist()),
        "subsets": rows,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
