import torch
import torch.nn as nn
import lightning as L
import numpy as np

from utils.validate import validate, calculate_acc_auc_f1 # Added calculate_acc_auc_f1
from utils.network_factory import get_model

 
class Trainer(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        # Add storage for test outputs (using a dictionary to handle multiple test dataloaders)
        self.test_step_outputs = {}
        self.criterion = nn.BCEWithLogitsLoss()


    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)['logits']
        loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch):
        x, y = batch
        logits = self.model(x)['logits']
        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)


    def on_validation_epoch_end(self):
        if not self.validation_step_outputs_preds: # Avoid errors if validation is skipped
            return
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(
                torch.float32).sigmoid().flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32).cpu().numpy()
        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)
        self.log('val_acc_epoch', acc, logger=True, sync_dist=True)
        self.log('val_ap_epoch', ap, logger=True, sync_dist=True)
        self.log('val_racc_epoch', r_acc, logger=True, sync_dist=True)
        self.log('val_facc_epoch', f_acc, logger=True, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()


    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx, dataloader_idx=0): # Add dataloader_idx
        x, y = batch
        logits = self.model(x)['logits']
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
