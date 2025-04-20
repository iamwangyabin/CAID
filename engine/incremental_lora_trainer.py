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

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if hasattr(self.model, 'current_stage_active_rank_count'):
             self.log('active_rank', float(self.model.current_stage_active_rank_count), on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        logits = outputs.get('current_logits')

        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs_preds or not self.validation_step_outputs_gts:
             print("Validation epoch end: No outputs collected.")
             self.log_dict({'val_ap': 0.0, 'val_acc': 0.0, 'val_ap_epoch': 0.0}, on_epoch=True, logger=True)
             return

        all_preds_tensor = torch.cat(self.validation_step_outputs_preds, 0)
        all_gts_tensor = torch.cat(self.validation_step_outputs_gts, 0)

        all_preds = all_preds_tensor.to(torch.float32).sigmoid().flatten().cpu().numpy()
        all_gts = all_gts_tensor.to(torch.float32).cpu().numpy()

        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)

        self.log_dict({
            'val_acc': acc,
            'val_ap': ap,
            'val_racc': r_acc,
            'val_facc': f_acc
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_ap_epoch', ap, on_epoch=True, logger=False, sync_dist=True)

        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()

    def configure_optimizers(self):
        if hasattr(self.model, 'set_trainable_stage') and callable(getattr(self.model, 'set_trainable_stage')):
             self.model.set_trainable_stage(self.current_stage_index)

        trainable_params_list = [p for p in self.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable_params_list)

        optimizer = hydra.utils.instantiate(self.opt.train.optimizer, params=trainable_params_list)
        scheduler_config = hydra.utils.instantiate(self.opt.train.scheduler, optimizer=optimizer)

        scheduler_dict = {
            "scheduler": scheduler_config,
            "interval": "step", # Or "epoch" depending on your scheduler logic
            "frequency": 1,
        }
        return [optimizer], [scheduler_dict]



    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        outputs = self(x)
        # Get specific stage_logits for testing as per user feedback and previous logic
        stage_key = f'stage_{dataloader_idx}' # dataloader_idx usually corresponds to the stage index in testing
        logits = outputs.get('stage_logits', {}).get(stage_key)

        if logits is None:
            # Attempt to fall back to current_logits if specific stage logit isn't available,
            # though ideally stage_logits should contain results for all tested stages.
            print(f"Warning: Logits for '{stage_key}' not found in test_step for dataloader {dataloader_idx}. Trying 'current_logits'.")
            logits = outputs.get('current_logits')
            if logits is None:
                 print(f"Error: No suitable logits found ('{stage_key}' or 'current_logits') in test_step for dataloader {dataloader_idx}. Skipping batch.")
                 return # Skip if no logits are found

        preds = logits.squeeze(1)
        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = {'preds': [], 'gts': []}
        self.test_step_outputs[dataloader_idx]['preds'].append(preds)
        self.test_step_outputs[dataloader_idx]['gts'].append(y)


    def on_test_epoch_end(self):
        log_dict = {}
        print("Calculating test metrics...")
        for dataloader_idx, outputs in self.test_step_outputs.items():
            if not outputs['preds'] or not outputs['gts']:
                 print(f"No outputs collected for test dataloader {dataloader_idx}. Skipping.")
                 continue
            try:
                all_preds_tensor = torch.cat(outputs['preds'], 0)
                all_gts_tensor = torch.cat(outputs['gts'], 0)

                all_preds = all_preds_tensor.to(torch.float32).sigmoid().flatten().cpu().numpy()
                all_gts = all_gts_tensor.to(torch.float32).cpu().numpy()

                r_acc, f_acc, acc, auc, f1, ap = calculate_acc_auc_f1(all_gts % 2, all_preds, threshold=0.5)

                prefix = f"{dataloader_idx}/"
                log_dict.update({
                    f"{prefix}r_acc": r_acc,
                    f"{prefix}f_acc": f_acc,
                    f"{prefix}acc": acc,
                    f"{prefix}auc": auc,
                    f"{prefix}f1": f1,
                    f"{prefix}ap": ap,
                })
                print(f"Metrics for dataloader {dataloader_idx}: AP={ap:.4f}, AUC={auc:.4f}, Acc={acc:.4f}")
            except Exception as e:
                print(f"Error calculating metrics for test dataloader {dataloader_idx}: {e}")

        self.test_results = log_dict
        self.test_step_outputs.clear()