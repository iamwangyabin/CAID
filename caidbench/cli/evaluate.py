from __future__ import annotations

import argparse
import json

from ..config import add_common_train_args, apply_overrides, load_config
from ..engine import Trainer
from ..utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CAIDBench checkpoint on the configured continual scenario")
    add_common_train_args(parser)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    trainer = Trainer(apply_overrides(load_config(args.config), args.override))
    try:
        ckpt = load_checkpoint(args.checkpoint, map_location=trainer.device)
        state = ckpt.get("model", ckpt)
        trainer.method.load_state_dict(state, strict=False)
        trainer.method.load_auxiliary_state_dict(ckpt.get("auxiliary"))
        row_index = max(len(trainer.scenario.tasks) - 1, 0)
        checkpoint_task_index = int(ckpt.get("task_index", row_index))
        after_task_name = trainer.scenario.tasks[min(max(checkpoint_task_index, 0), row_index)].name if trainer.scenario.tasks else "checkpoint"
        eval_payload: dict[str, float | int] = {}
        eval_rows = []
        for j in range(len(trainer.scenario.tasks)):
            loader = trainer.dataloader(j, "test", shuffle=False)
            metrics = trainer.evaluate_loader(loader)
            trainer.metric_matrix.update(row_index, j, metrics["acc"], metrics["auc"], metrics["ap"], metrics["f1"])
            record = {
                "after_task": checkpoint_task_index,
                "after_task_name": after_task_name,
                "eval_task": j,
                "eval_task_name": trainer.scenario.tasks[j].name,
                "acc": metrics["acc"],
                "auc": metrics["auc"],
                "ap": metrics["ap"],
                "f1": metrics["f1"],
                "ece": metrics["ece"],
                "num_samples": metrics["num_samples"],
            }
            trainer.eval_records.append(record)
            eval_rows.append(record)
        eval_payload.update(
            {
                "eval/average_accuracy": trainer.metric_matrix.average_accuracy(train_index=row_index, kind="acc"),
                "eval/average_auc": trainer.metric_matrix.average_accuracy(train_index=row_index, kind="auc"),
                "eval/average_ap": trainer.metric_matrix.average_accuracy(train_index=row_index, kind="ap"),
                "eval/average_f1": trainer.metric_matrix.average_accuracy(train_index=row_index, kind="f1"),
                "eval/after_task": checkpoint_task_index,
            }
        )
        trainer._log_eval_table(eval_rows, step=checkpoint_task_index)
        trainer.log_metrics(eval_payload, step=checkpoint_task_index)
        summary = trainer._write_outputs()
        trainer.log_metrics(
            {
                "summary/average_accuracy": summary["average_accuracy"],
                "summary/average_forgetting": summary["average_forgetting"],
                "summary/average_auc": summary["average_auc"],
                "summary/auc_forgetting": summary["auc_forgetting"],
                "summary/average_ap": summary["average_ap"],
                "summary/ap_forgetting": summary["ap_forgetting"],
                "summary/average_f1": summary["average_f1"],
                "summary/f1_forgetting": summary["f1_forgetting"],
            }
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        trainer.experiment.finish()


if __name__ == "__main__":
    main()
