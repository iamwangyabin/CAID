import os
import hydra
import argparse
import wandb
import datetime
import copy

import torch
import torch.nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import engine
import data
import networks
from utils.common import load_config_with_cli, archive_files, seed_everything
from utils.dataloader import build_train_val_dataloader, build_test_dataloader

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

    wandb_logger = WandbLogger(name=today_str, project='ContinualAIDetect',
                               job_type='train', group=conf.name, save_dir=base_log_dir)

    if os.getenv("LOCAL_RANK", '0') == '0':
        archive_files(today_str, exclude_dirs=['logs', 'wandb', '.git', 'exp_results', '__pycache__'])

    print("Preparing test dataloaders...")
    test_loaders_dict = {}
    test_loader_conf = conf.datasets.test 
    test_trsf = conf.datasets.test.trsf  
    for i, test_source_conf in enumerate(conf.datasets.test.source):
        test_loader = build_test_dataloader(test_source_conf, test_loader_conf, test_trsf)
        test_loaders_dict[test_source_conf.benchmark_name] = test_loader


    last_stage_checkpoint = None
    model = None
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
        else:
            model = hydra.utils.get_class(conf.train.pipeline).load_from_checkpoint(last_stage_checkpoint, opt=conf) 

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
            logger=wandb_logger,
            max_epochs=conf.train.train_epochs,
            accelerator="gpu",
            devices=conf.train.gpu_ids, 
            callbacks=[stage_checkpoint_callback],
            check_val_every_n_epoch=conf.train.check_val_every_n_epoch,  
            precision="16",
            log_every_n_steps=100,
        )

        trainer.fit(model=model, train_dataloaders=stage_train_loader, val_dataloaders=stage_val_loader)

        trainer.test(model=model, dataloaders=list(test_loaders_dict.values()))
        test_results = model.test_results
# {'0/r_acc': 0.9856262833675564, '0/f_acc': 0.12114989733059549, '0/acc': 0.553388090349076, '0/auc': np.float64(0.7602173977206126), '0/f1': 0.21338155515370705, '0/ap': np.float64(0.7515994398274075), '1/r_acc': 0.9955, '1/f_acc': 0.5065, '1/acc': 0.751, '1/auc': np.float64(0.9401407500000001), '1/f1': 0.6704169424222369, '1/ap': np.float64(0.9483890705884956)}

        import pdb; pdb.set_trace()
        metrics_to_log_this_stage = {}
        for test_idx, test_name in enumerate(test_loaders_dict.keys()):
            single_test_result = stage_test_results_list[test_idx] # 获取对应测试集的结果字典
            print(f"  Test Set: {test_name}")
            if isinstance(single_test_result, dict):
                for metric_key, metric_value in single_test_result.items():
                    # 清理 Lightning 可能添加的前缀（虽然我们的 MockPipeline 没加，但以防万一）
                    clean_metric_key = metric_key.split('/')[-1]
                    # 构建 WandB 使用的 key: stage_{i}/test_{TestSetName}/{MetricName}
                    wandb_key = f"stage_{i}/test_{test_name}/{clean_metric_key}"
                    metrics_to_log_this_stage[wandb_key] = metric_value
        # for test_name, test_loader in test_loaders_dict.items():
        #     print(f"Testing on: {test_name}")
        #     single_test_result = test_results[0]
        #     all_stage_test_results[test_name] = single_test_result
        #     for key, value in single_test_result.items():
        #         wandb_logger.log_metrics({f"{i}/{test_name}_{key}": value})


        last_stage_checkpoint = stage_checkpoint_callback.best_model_path




if __name__ == '__main__':
    run_incremental_training()

