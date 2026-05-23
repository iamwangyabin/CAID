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
            eval_payload: dict[str, float | int] = {}
            for j in range(i + 1):
                loader = trainer.dataloader(j, "test", shuffle=False)
                metrics = trainer.evaluate_loader(loader)
                trainer.metric_matrix.update(i, j, metrics["acc"], metrics["auc"])
                trainer.eval_records.append(
                    {
                        "after_task": i,
                        "after_task_name": _task.name,
                        "eval_task": j,
                        "eval_task_name": trainer.scenario.tasks[j].name,
                        "acc": metrics["acc"],
                        "auc": metrics["auc"],
                        "ece": metrics["ece"],
                    }
                )
                eval_payload[f"eval/task_{j}/acc"] = metrics["acc"]
                eval_payload[f"eval/task_{j}/auc"] = metrics["auc"]
                eval_payload[f"eval/task_{j}/ece"] = metrics["ece"]
            eval_payload.update(
                {
                    "eval/average_accuracy": trainer.metric_matrix.average_accuracy(train_index=i, kind="acc"),
                    "eval/average_auc": trainer.metric_matrix.average_accuracy(train_index=i, kind="auc"),
                    "eval/after_task": i,
                }
            )
            trainer.log_metrics(eval_payload, step=i)
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
