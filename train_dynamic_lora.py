import os
import hydra
import argparse
import wandb
import datetime
import copy

import torch
import torch.nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from train import IncrementalLearningPipeline
from utils.callbacks import RankIncrementCallback

import engine
import data
import networks


class DynamicLoraPipeline(IncrementalLearningPipeline):

    def _train_stage(self, stage_index, dataset_conf):
        print(f"--- Preparing Dynamic LoRA Stage {stage_index + 1} ---")
        step_log_dir = os.path.join(self.base_log_dir, f"stage_{stage_index}")
        os.makedirs(step_log_dir, exist_ok=True)

        stage_train_loader, stage_val_loader = self._prepare_train_val_loaders(dataset_conf, stage_index)

        print(f"Loading model for stage {stage_index + 1}...")
        if self.last_stage_checkpoint is None:
            self.model = hydra.utils.get_class(self.conf.train.pipeline)(opt=self.conf)
            self.model.cumulative_step = self.cumulative_step
        else:
            self.model = hydra.utils.get_class(self.conf.train.pipeline).load_from_checkpoint(
                self.last_stage_checkpoint,
                opt=self.conf,
                stage_index=stage_index,
                map_location='cpu' 
            )
            # self.model = hydra.utils.get_class(self.conf.train.pipeline)(opt=self.conf, stage_index=stage_index)

            self.model.cumulative_step = self.cumulative_step 

        stage_checkpoint_callback = ModelCheckpoint(
            monitor='val_ap_epoch',
            dirpath=step_log_dir,
            filename='best',
            save_top_k=1,
            mode='max',
            save_last=True,
            save_weights_only=False
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        rank_increase_interval_epochs = self.conf.train.get('rank_increase_interval_epochs', 1)
        rank_increment_amount = self.conf.train.get('rank_increment_amount', 1)    # Rank 增长量
        rank_increment_callback = RankIncrementCallback(
            increase_interval_epochs=rank_increase_interval_epochs,
            increment_amount=rank_increment_amount
        )

        callbacks = [stage_checkpoint_callback, lr_monitor, rank_increment_callback]

        trainer = L.Trainer(
            max_epochs=self.conf.train.train_epochs,
            accelerator="gpu",
            devices=[int(x) for x in self.conf.train.gpu_ids],
            callbacks=callbacks,
            check_val_every_n_epoch=self.conf.train.check_val_every_n_epoch,
            precision="16-mixed",
            log_every_n_steps=self.conf.train.get('log_every_n_steps', 50),
        )

        print(f"Starting training for stage {stage_index + 1}...")

        trainer.fit(
            model=self.model,
            train_dataloaders=stage_train_loader,
            val_dataloaders=stage_val_loader
        )
        print(f"Finished training for stage {stage_index + 1}.")


        self.last_stage_checkpoint = stage_checkpoint_callback.best_model_path
        if not self.last_stage_checkpoint or not os.path.exists(self.last_stage_checkpoint):
             print("Warning: No best model checkpoint found or path invalid. Using last checkpoint.")
             self.last_stage_checkpoint = stage_checkpoint_callback.last_model_path
             if not self.last_stage_checkpoint or not os.path.exists(self.last_stage_checkpoint):
                  raise FileNotFoundError("Neither best nor last checkpoint path is valid after training stage.")

        print(f"Stage {stage_index + 1} training finished. Checkpoint for next stage: {self.last_stage_checkpoint}")
        return trainer

    def _test_stage(self, trainer, stage_index):
        print(f"Loading model from {self.last_stage_checkpoint} for testing.")
        model_to_test = hydra.utils.get_class(self.conf.train.pipeline).load_from_checkpoint(
                self.last_stage_checkpoint,
                opt=self.conf,
                stage_index=stage_index 
            )
        trainer.test(model=model_to_test, dataloaders=list(self.test_loaders_dict.values()), verbose=False)

        test_results = model_to_test.test_results
        mapped_results = {}
        for k, v in test_results.items():
            parts = k.split('/')
            if len(parts) > 1 and parts[0] in self.id_to_benchmark:
                benchmark_name = self.id_to_benchmark[parts[0]]
                metric_name = '/'.join(parts[1:])
                mapped_results[f"stage_{stage_index+1}/{benchmark_name}/{metric_name}"] = v
            else:
                mapped_results[f"stage_{stage_index+1}/{k}"] = v
                print(f"Warning: Could not map test result key '{k}' using id_to_benchmark.")

        session_final_step = self.cumulative_step + trainer.global_step
        print(f"Logging test results for Stage {stage_index+1} at step {session_final_step}: {mapped_results}")
        wandb.log(mapped_results, step=session_final_step)

        self.cumulative_step += trainer.global_step


if __name__ == '__main__':

    pipeline = DynamicLoraPipeline()
    pipeline.run()
