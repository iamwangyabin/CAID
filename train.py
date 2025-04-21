import os
import hydra
import argparse
import wandb
import datetime
import copy

import torch
import torch.nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor # Import LearningRateMonitor

import engine
import data
import networks
from utils.common import load_config_with_cli, archive_files, seed_everything
from utils.dataloader import build_train_val_dataloader, build_test_dataloader

class IncrementalLearningPipeline:
    """
    Manages the incremental learning process, encapsulating configuration,
    logging, data loading, training stages, and testing.
    """
    def __init__(self):
        """
        Initializes the pipeline by parsing arguments, loading configuration,
        and setting up the environment. Logging and data loading are deferred
        to the `run` method.
        """
        self.args, self.cfg_args = self._parse_args()
        self.conf = self._load_config()
        self._initialize_environment()
        self.base_log_dir = None
        self.today_str = None
        self.test_loaders_dict = {}
        self.id_to_benchmark = {}
        self.model = None
        self.last_stage_checkpoint = None
        self.cumulative_step = 0

    def _parse_args(self):
        """Parses command-line arguments."""
        parser = argparse.ArgumentParser(description='Incremental Training')
        parser.add_argument('--cfg', type=str, default='cfgs/incremental_rine.yaml', required=False,
                            help='Path to the incremental configuration file.')
        return parser.parse_known_args()

    def _load_config(self):
        """Loads and instantiates the configuration using Hydra."""
        conf = load_config_with_cli(self.args.cfg, args_list=self.cfg_args)
        return hydra.utils.instantiate(conf)

    def _initialize_environment(self):
        """Sets random seeds and PyTorch configurations."""
        seed_everything(self.conf.train.seed)
        torch.set_float32_matmul_precision('high')

    def _setup_logging(self):
        """Sets up the logging directory and initializes wandb."""
        # Ensure this is called only once per run
        if self.base_log_dir is not None:
            return self.base_log_dir, self.today_str

        print("Setting up logging...")
        today_str = self.conf.name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
        base_log_dir = os.path.join('logs', today_str)
        os.makedirs(base_log_dir, exist_ok=True)

        wandb.init(name=today_str, project='ContinualAIDetect',
                   job_type='train', group=self.conf.name, dir=base_log_dir)

        # Archive code only on the main process
        if os.getenv("LOCAL_RANK", '0') == '0':
            archive_files(today_str, exclude_dirs=['logs', 'wandb', '.git', 'exp_results', '__pycache__'])

        print(f"Logging initialized. Log directory: {base_log_dir}")
        return base_log_dir, today_str

    def _prepare_test_loaders(self):
        """Builds and returns test dataloaders for all specified benchmarks."""
        # Ensure this is called only once per run
        if self.test_loaders_dict:
             return self.test_loaders_dict, self.id_to_benchmark

        print("Preparing test dataloaders...")
        test_loaders_dict = {}
        id_to_benchmark = {}
        test_loader_conf = self.conf.datasets.test
        test_trsf = self.conf.datasets.test.trsf
        for i, test_source_conf in enumerate(self.conf.datasets.test.source):
            test_loader = build_test_dataloader(test_source_conf, test_loader_conf, test_trsf)
            test_loaders_dict[test_source_conf.benchmark_name] = test_loader
            id_to_benchmark[str(i)] = test_source_conf.benchmark_name
        print(f"Prepared test loaders for benchmarks: {list(test_loaders_dict.keys())}")
        return test_loaders_dict, id_to_benchmark

    def _prepare_train_val_loaders(self, dataset_conf, stage_index):
        """Builds train and validation dataloaders for a specific stage."""
        print(f"Building train/val dataloaders for stage {stage_index + 1}...")
        stage_train_loader, stage_val_loader = build_train_val_dataloader(
            source_list=[dataset_conf],
            loader_conf=self.conf.datasets.train,
            trsf=self.conf.datasets.train.trsf
        )
        return stage_train_loader, stage_val_loader

    def _train_stage(self, stage_index, dataset_conf):
        """Handles the training process for a single incremental stage."""
        print(f"--- Preparing Stage {stage_index + 1} ---")
        step_log_dir = os.path.join(self.base_log_dir, f"step_{stage_index+1}")
        os.makedirs(step_log_dir, exist_ok=True)

        stage_train_loader, stage_val_loader = self._prepare_train_val_loaders(dataset_conf, stage_index)

        print(f"Loading model for stage {stage_index + 1}...")
        if self.last_stage_checkpoint is None:
            print("Initializing new model.")
            self.model = hydra.utils.get_class(self.conf.train.pipeline)(opt=self.conf)
            self.model.cumulative_step = self.cumulative_step
        else:
            print(f"Loading model from checkpoint: {self.last_stage_checkpoint}")
            # Ensure the checkpoint file exists before attempting to load
            if not os.path.exists(self.last_stage_checkpoint):
                 print(f"Error: Checkpoint file not found at {self.last_stage_checkpoint}. Cannot load model.")
                 raise FileNotFoundError(f"Checkpoint file not found: {self.last_stage_checkpoint}")

            self.model = hydra.utils.get_class(self.conf.train.pipeline).load_from_checkpoint(
                self.last_stage_checkpoint,
                opt=self.conf,
                map_location='cpu' # Load to CPU first
            )
            self.model.cumulative_step = self.cumulative_step # Set after loading

        stage_checkpoint_callback = ModelCheckpoint(
            monitor='val_ap_epoch',
            dirpath=step_log_dir,
            filename='best-{epoch}-{val_ap_epoch:.4f}',
            save_top_k=1,
            mode='max',
            save_last=True,
            save_weights_only=False
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        print(f"Initializing Trainer for stage {stage_index + 1}...")
        trainer = L.Trainer(
            max_epochs=self.conf.train.train_epochs,
            accelerator="gpu",
            devices=[int(x) for x in self.conf.train.gpu_ids],
            callbacks=[stage_checkpoint_callback, lr_monitor],
            check_val_every_n_epoch=self.conf.train.check_val_every_n_epoch,
            precision="16-mixed",
            log_every_n_steps=self.conf.train.get('log_every_n_steps', 50),
        )

        print(f"Starting training for stage {stage_index + 1}...")
        trainer.fit(model=self.model, train_dataloaders=stage_train_loader, val_dataloaders=stage_val_loader)

        self.last_stage_checkpoint = stage_checkpoint_callback.best_model_path
        if not self.last_stage_checkpoint or not os.path.exists(self.last_stage_checkpoint):
             print("Warning: No best model checkpoint found or path invalid. Using last checkpoint.")
             self.last_stage_checkpoint = stage_checkpoint_callback.last_model_path
             if not self.last_stage_checkpoint or not os.path.exists(self.last_stage_checkpoint):
                  print("Error: Last checkpoint path is also invalid.")
                  raise FileNotFoundError("Neither best nor last checkpoint path is valid.")

        print(f"Stage {stage_index + 1} training finished. Checkpoint for next stage: {self.last_stage_checkpoint}")
        return trainer

    def _test_stage(self, trainer, stage_index):
        print(f"--- Testing after Stage {stage_index + 1} ---")
        model_to_test = hydra.utils.get_class(self.conf.train.pipeline).load_from_checkpoint(
                self.last_stage_checkpoint,
                opt=self.conf
            )

        trainer.test(model=model_to_test, dataloaders=list(self.test_loaders_dict.values()))

        if hasattr(model_to_test, 'test_results') and model_to_test.test_results:
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
        else:
            print("No test results found on the model to log after testing.")

        self.cumulative_step += trainer.global_step
        print(f"Cumulative step updated to: {self.cumulative_step}")

    def run(self):
        """Executes the full incremental learning pipeline stage by stage."""
        print("Initializing pipeline run...")
        # Setup logging and test loaders at the beginning of the run
        self.base_log_dir, self.today_str = self._setup_logging()
        self.test_loaders_dict, self.id_to_benchmark = self._prepare_test_loaders()

        print("Starting incremental learning stages...")
        try:
            for i, dataset_conf in enumerate(self.conf.datasets.train.source):
                print(f"\n{'='*20} Stage {i+1} {'='*20}")
                trainer = self._train_stage(i, dataset_conf)
                self._test_stage(trainer, i)
                print(f"{'='*20} Stage {i+1} Complete {'='*20}")

            print("\nIncremental learning finished successfully.")

        except Exception as e:
            print(f"\n{'!'*20} Pipeline interrupted due to error: {e} {'!'*20}")
            wandb.log({"pipeline_error": str(e)})
            raise

        finally:
            print("Finishing wandb run.")
            if wandb.run: # Check if wandb was initialized before finishing
                 wandb.finish()


if __name__ == '__main__':
    pipeline = IncrementalLearningPipeline()
    pipeline.run()
