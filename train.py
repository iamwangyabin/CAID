from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from caidbench.cli.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
