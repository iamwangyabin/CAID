import os
import hydra
import argparse
import wandb
import datetime
import copy

import torch
import torch.nn
import lightning as L
# from lightning.pytorch.loggers import WandbLogger # Remove WandbLogger import
from lightning.pytorch.callbacks import ModelCheckpoint

import engine
import data
import networks
from utils.common import load_config_with_cli, archive_files, seed_everything
from utils.dataloader import build_train_val_dataloader, build_test_dataloader
# Remove StepOffsetCallback import

def run_incremental_training():
    parser = argparse.ArgumentParser(description='Incremental Training')
    parser.add_argument('--cfg', type=str, default='cfgs/incremental_rine.yaml', required=False, # Made default for convenience
                        help='Path to the incremental configuration file.')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    seed_everything(conf.train.seed)
    torch.set_float32_matmul_precision('high')

    today_str = conf.name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    base_log_dir = os.path.join('logs', today_str)
    os.makedirs(base_log_dir, exist_ok=True)

    wandb.init(name=today_str, project='ContinualAIDetect',
               job_type='train', group=conf.name, dir=base_log_dir)

    if os.getenv("LOCAL_RANK", '0') == '0':
        archive_files(today_str, exclude_dirs=['logs', 'wandb', '.git', 'exp_results', '__pycache__'])

    print("Preparing test dataloaders...")
    test_loaders_dict = {}
    id_to_benchmark = {}
    test_loader_conf = conf.datasets.test 
    test_trsf = conf.datasets.test.trsf  
    for i, test_source_conf in enumerate(conf.datasets.test.source):
        test_loader = build_test_dataloader(test_source_conf, test_loader_conf, test_trsf)
        test_loaders_dict[test_source_conf.benchmark_name] = test_loader
        id_to_benchmark[str(i)] = test_source_conf.benchmark_name


    last_stage_checkpoint = None
    model = None
    cumulative_step = 0
    for i, dataset_conf in enumerate(conf.datasets.train.source):
        step_log_dir = os.path.join(base_log_dir, f"step_{i+1}")
        os.makedirs(step_log_dir, exist_ok=True)

        stage_train_loader, stage_val_loader = build_train_val_dataloader(
            source_list=[dataset_conf], 
            loader_conf=conf.datasets.train,
            trsf=conf.datasets.train.trsf
        )

        if last_stage_checkpoint is None:
            model = hydra.utils.get_class(conf.train.pipeline)(opt=conf)
            model.cumulative_step = cumulative_step 
        else:
            model = hydra.utils.get_class(conf.train.pipeline).load_from_checkpoint(
                last_stage_checkpoint,
                opt=conf
            )
            model.cumulative_step = cumulative_step # Set after loading from checkpoint

        stage_checkpoint_callback = ModelCheckpoint(
            monitor='val_ap_epoch',
            dirpath=step_log_dir,
            filename='best',
            save_top_k=1,
            mode='max',
            save_last=False,
            save_weights_only=True
        )

        trainer = L.Trainer(
            max_epochs=conf.train.train_epochs,
            accelerator="gpu",
            devices=[int(x) for x in conf.train.gpu_ids],
            callbacks=[stage_checkpoint_callback],
            check_val_every_n_epoch=conf.train.check_val_every_n_epoch,
            precision="16-mixed", # Use recommended precision
        )

        trainer.fit(model=model, train_dataloaders=stage_train_loader, val_dataloaders=stage_val_loader)


        trainer.test(model=model, dataloaders=list(test_loaders_dict.values()))
        test_results = model.test_results
        test_results = {f"{id_to_benchmark[k.split('/')[0]]}/{k.split('/', 1)[1]}": v for k, v in test_results.items()}
        session_final_step = cumulative_step + trainer.global_step
        wandb.log(test_results, step=session_final_step)

        last_stage_checkpoint = stage_checkpoint_callback.best_model_path
        cumulative_step += trainer.global_step




if __name__ == '__main__':
    run_incremental_training()

