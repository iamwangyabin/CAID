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
                map_location='cpu' 
            )
            self.model.cumulative_step = self.cumulative_step 




        checkpoint_monitor_key = self.conf.train.get('checkpoint_monitor', 'val_ap') # 监控指标
        checkpoint_mode = self.conf.train.get('checkpoint_mode', 'max')             # 监控模式 (max 或 min)
        stage_checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_monitor_key,
            dirpath=step_log_dir,
            filename=f'stage_{stage_index}_best_{{epoch:02d}}-{{{checkpoint_monitor_key}:.4f}}', # 文件名格式
            save_top_k=1,             
            mode=checkpoint_mode,
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
        # 如果最佳检查点路径无效 (例如验证集性能没有提升)，则尝试使用最后一个检查点
        if not self.last_stage_checkpoint or not os.path.exists(self.last_stage_checkpoint):
             print("Warning: No best model checkpoint found or path invalid. Using last checkpoint.")
             self.last_stage_checkpoint = stage_checkpoint_callback.last_model_path
             # 如果最后一个检查点路径也无效，则增量学习无法继续，必须报错
             if not self.last_stage_checkpoint or not os.path.exists(self.last_stage_checkpoint):
                  raise FileNotFoundError("Neither best nor last checkpoint path is valid after training stage.")

        print(f"Stage {stage_index + 1} training finished. Checkpoint for next stage: {self.last_stage_checkpoint}")
        return trainer

    def _test_stage(self, trainer, stage_index):

        print(f"--- Testing after Dynamic LoRA Stage {stage_index + 1} ---")

        model_to_test = None
        # 必须有有效的检查点才能进行测试
        if self.last_stage_checkpoint and os.path.exists(self.last_stage_checkpoint):
            print(f"Loading model from {self.last_stage_checkpoint} for testing.")
            pipeline_class_path = self.conf.train.pipeline._target_
            pipeline_class = hydra.utils.get_class(pipeline_class_path)
            # 加载模型用于测试 (加载失败会直接报错)
            # **Dynamic LoRA 特定**: 如果测试逻辑需要，传递 stage_index
            model_to_test = pipeline_class.load_from_checkpoint(
                 self.last_stage_checkpoint,
                 opt=self.conf,
                 stage_index=stage_index # 传递阶段索引，以防测试逻辑需要
             )
        else:
            # 如果没有检查点（理论上不应发生，因为 _train_stage 会检查），则跳过测试
            print(f"Critical Error: Checkpoint {self.last_stage_checkpoint} not found before testing. Testing skipped.")
            return

        # 执行测试 (测试失败会直接报错)
        trainer.test(model=model_to_test, dataloaders=list(self.test_loaders_dict.values()), verbose=False)

        # 处理并记录测试结果
        if hasattr(model_to_test, 'test_results') and model_to_test.test_results:
            test_results = model_to_test.test_results
            mapped_results = {}
            # 格式化日志键名
            for k, v in test_results.items():
                parts = k.split('/')
                # 假设原始键名格式为 "dataloader_idx/metric_name"
                if len(parts) > 1 and parts[0] in self.id_to_benchmark:
                    benchmark_name = self.id_to_benchmark[parts[0]] # 获取数据集名称
                    metric_name = '/'.join(parts[1:])               # 获取指标名称
                    # **Dynamic LoRA 特定**: 添加阶段前缀
                    mapped_results[f"stage_{stage_index}/test/{benchmark_name}/{metric_name}"] = v
                else:
                    # 如果键名格式不符合预期，保留原始键名并添加前缀
                    mapped_results[f"stage_{stage_index}/test/unknown/{k}"] = v
                    print(f"Warning: Could not map test result key '{k}' using id_to_benchmark.")

            # 计算当前总步数用于日志记录
            session_final_step = self.cumulative_step + trainer.global_step
            print(f"Logging test results for Stage {stage_index+1} at step {session_final_step}: {mapped_results}")

            # 使用 Trainer 配置的 Logger 记录指标 (例如 Wandb)
            if trainer.logger:
                trainer.logger.log_metrics(mapped_results, step=session_final_step) # 记录失败会直接报错
            else:
                print("Trainer logger not available, skipping metric logging.")
        else:
            print("No test results found on the model to log after testing.")


if __name__ == '__main__':

    pipeline = DynamicLoraPipeline()
    pipeline.run()
