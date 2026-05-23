from __future__ import annotations

import argparse
import json

import torch

from ..config import load_config
from ..engine import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CAIDBench checkpoint on the configured continual scenario")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    trainer = Trainer(load_config(args.config))
    try:
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        state = ckpt.get("model", ckpt)
        trainer.method.load_state_dict(state, strict=False)
        for i, _task in enumerate(trainer.scenario.tasks):
            for j in range(i + 1):
                loader = trainer.dataloader(j, "test", shuffle=False)
                metrics = trainer.evaluate_loader(loader)
                trainer.metric_matrix.update(i, j, metrics["acc"], metrics["auc"])
                trainer.log_metrics(
                    {
                        f"eval/task_{j}/acc": metrics["acc"],
                        f"eval/task_{j}/auc": metrics["auc"],
                        f"eval/task_{j}/ece": metrics["ece"],
                        "eval/after_task": i,
                        "eval/on_task": j,
                    }
                )
        summary = trainer._write_outputs()
        trainer.log_metrics(
            {
                "summary/average_accuracy": summary["average_accuracy"],
                "summary/average_forgetting": summary["average_forgetting"],
                "summary/average_auc": summary["average_auc"],
                "summary/auc_forgetting": summary["auc_forgetting"],
            }
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        trainer.experiment.finish()


if __name__ == "__main__":
    main()
