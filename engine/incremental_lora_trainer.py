import torch
import torch.nn as nn
import wandb
import lightning as L
import numpy as np
import hydra

from omegaconf import OmegaConf 

import torch.optim as optim

from utils.network_factory import get_model
from utils.validate import validate, calculate_acc_auc_f1

class IncrementalLoRATrainer(L.LightningModule):
    def __init__(self, opt, stage_index=0):
        super().__init__()
        self.opt = opt
        self.current_stage_index = stage_index
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.test_step_outputs = {}
        self.criterion = nn.BCEWithLogitsLoss()

        self.test_results = {}
        self.cumulative_step = 0

    def on_train_start(self):
        initial_rank = self.opt.train.get('initial_active_rank', 1)
        self.model.start_new_stage(self.current_stage_index, initial_active_rank=initial_rank)
        self.model.train()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        logits = outputs.get('current_logits')

        loss = self.criterion(logits.squeeze(1), (y % 2).to(logits.dtype))
        wandb.log({"train_loss": loss}, step=self.trainer.global_step + self.cumulative_step)
        if hasattr(self.model, 'current_stage_active_rank_count'):
            wandb.log({"active_rank": float(self.model.current_stage_active_rank_count)}, step=self.trainer.global_step + self.cumulative_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        logits = outputs.get('current_logits')

        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        all_preds_tensor = torch.cat(self.validation_step_outputs_preds, 0)
        all_gts_tensor = torch.cat(self.validation_step_outputs_gts, 0)

        all_preds = all_preds_tensor.to(torch.float32).sigmoid().flatten().cpu().numpy()
        all_gts = all_gts_tensor.to(torch.float32).cpu().numpy()
        current_step = self.trainer.global_step + self.cumulative_step
        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)
        wandb.log({
            'val_acc_epoch': acc,
            'val_ap_epoch': ap,
            'val_racc_epoch': r_acc,
            'val_facc_epoch': f_acc
        }, step=current_step)
        self.log('val_ap_epoch', ap, logger=False, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()


    def configure_optimizers(self):
        if hasattr(self.model, 'set_trainable_stage') and callable(getattr(self.model, 'set_trainable_stage')):
             self.model.set_trainable_stage(self.current_stage_index)

        trainable_params_list = [p for p in self.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable_params_list)

        optimizer = self.opt.train.optimizer(params=trainable_params_list)
        scheduler = self.opt.train.scheduler(optimizer)

        return [optimizer], [scheduler]


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.model(x)['current_logits']
        preds = logits.squeeze(1)
        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = {'preds': [], 'gts': []}
        self.test_step_outputs[dataloader_idx]['preds'].append(preds)
        self.test_step_outputs[dataloader_idx]['gts'].append(y)

    def on_test_epoch_end(self):
        log_dict = {}
        for dataloader_idx, outputs in self.test_step_outputs.items():
            all_preds = torch.cat(outputs['preds'], 0).to(
                torch.float32).sigmoid().flatten().cpu().numpy()
            all_gts = torch.cat(outputs['gts'], 0).to(torch.float32).cpu().numpy()
            r_acc, f_acc, acc, auc, f1, ap = calculate_acc_auc_f1(all_gts % 2, all_preds, 0.5) 
            prefix = f"{dataloader_idx}/"
            log_dict.update({
                f"{prefix}r_acc": r_acc,
                f"{prefix}f_acc": f_acc,
                f"{prefix}acc": acc,
                f"{prefix}auc": auc,
                f"{prefix}f1": f1,
                f"{prefix}ap": ap,
            })
        self.test_results = log_dict
        return log_dict
    