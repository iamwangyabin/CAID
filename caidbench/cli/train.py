from __future__ import annotations

import argparse
import json
import os

from ..config import add_common_train_args, apply_overrides, load_config
from ..engine import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CAIDBench continual detector")
    add_common_train_args(parser)
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    summary = Trainer(cfg).run()
    if int(os.environ.get("RANK", "0") or 0) == 0:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
