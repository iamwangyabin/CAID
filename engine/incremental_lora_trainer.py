import torch
import torch.nn as nn
import wandb
import lightning as L
import numpy as np
import hydra # Import hydra for instantiation

from utils.validate import validate, calculate_acc_auc_f1 # Added calculate_acc_auc_f1
# Import the specific creator function for the dynamic LoRA model
from networks.parallel_dynamic_lora_vit import create_parallel_dynamic_lora_vit
import torch.optim as optim # Import optim

# Rename the class to avoid conflicts and indicate its purpose
class IncrementalLoRATrainer(L.LightningModule):
    # Add stage_index to constructor
    def __init__(self, opt, stage_index=0):
        super().__init__()
        # self.save_hyperparameters(opt) # Causes issues with Hydra/OmegaConf sometimes, use self.opt
        self.opt = opt
        self.current_stage_index = stage_index # Store current stage index

        # --- Model Creation using the specific creator ---
        # Assuming model config is within opt, e.g., opt.model
        model_conf = self.opt.get('model', self.opt) # Adjust based on your config structure
        try:
            self.model = create_parallel_dynamic_lora_vit(
                model_name=model_conf.name,
                pretrained=model_conf.get('pretrained', True),
                num_classes=model_conf.num_classes,
                num_loras=model_conf.get('num_loras', 1),
                max_rank_potential=model_conf.get('max_rank_potential', 8),
                rank_dropout_p=model_conf.get('rank_dropout_p', 0.0),
                freeze_base=model_conf.get('freeze_base', True)
                # Pass other relevant args from model_conf if needed
            )
            print(f"Successfully created ParallelDynamicLoRA_ViT_timm for stage {stage_index}")
        except AttributeError as e:
             raise AttributeError(f"Missing required model configuration in 'opt.model': {e}. "
                                f"Ensure name, num_classes, num_loras, max_rank_potential are defined.")
        except Exception as e: # Catch other potential errors during creation
             print(f"Error creating dynamic LoRA model: {e}")
             raise e


        # --- Other initializations ---
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.test_step_outputs = {}
        # Define criterion based on config if needed, otherwise default
        # Use get to provide default if criterion config is missing
        criterion_conf = self.opt.train.get('criterion', {'_target_': 'torch.nn.BCEWithLogitsLoss'})
        try:
            self.criterion = hydra.utils.instantiate(criterion_conf)
            print(f"Criterion initialized: {self.criterion.__class__.__name__}")
        except Exception as e:
            print(f"Error instantiating criterion from config: {e}. Falling back to BCEWithLogitsLoss.")
            self.criterion = nn.BCEWithLogitsLoss()

        # self.cumulative_step = 0 # Seems less relevant with per-stage trainer.fit
        self.test_results = {} # Initialize test_results

    # Add on_train_start hook
    def on_train_start(self):
        """ Called before the training loop for each `trainer.fit` call. """
        # Determine initial rank for the stage (e.g., always 1 or from config)
        initial_rank = self.opt.train.get('initial_active_rank', 1)
        print(f"Trainer: on_train_start for stage {self.current_stage_index}. Initializing stage in model with rank {initial_rank}.")
        # Initialize the stage in the model
        if hasattr(self.model, 'start_new_stage'):
            self.model.start_new_stage(self.current_stage_index, initial_active_rank=initial_rank)
        else:
            print("Warning: Model does not have 'start_new_stage' method.")
        # Ensure model is in training mode
        self.model.train()

    def forward(self, x):
        # The dynamic LoRA model returns a dict
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # Adapt based on your data format
        outputs = self(x)
        logits = outputs.get('aggregated_lora_logit') # Use .get() for safety
        # lora_features = outputs.get('lora_features') # Also use .get() if needed

        if logits is None:
            # Logits might be None if no LoRA paths are active or configured
            # Handle this case: maybe return None, or calculate loss based on base_logits?
            # For now, let's log a warning and return None to skip the step.
            print(f"Warning: 'aggregated_lora_logit' not found or is None in training_step for stage {self.current_stage_index}. Skipping batch.")
            # Optionally log base_logits loss if available and desired:
            # base_logits = outputs.get('base_logits')
            # if base_logits is not None:
            #     loss = self.criterion(base_logits.squeeze(1), (y % 2).to(base_logits.dtype))
            #     self.log('train_base_loss', loss, on_step=True, on_epoch=True, logger=True)
            # else:
            #     print("Warning: Both aggregated_lora_logit and base_logits are None.")
            return None # Skip optimizer step for this batch

        # Ensure target dtype matches logits dtype
        loss = self.criterion(logits.squeeze(1), (y % 2).to(logits.dtype))

        # Log using Lightning's built-in logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log current active rank if model has the attribute
        if hasattr(self.model, 'current_stage_active_rank_count'):
             self.log('active_rank', float(self.model.current_stage_active_rank_count), on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        logits = outputs.get('aggregated_lora_logit') # Use .get() for safety

        if logits is None:
            # Logits might be None if no LoRA paths are active or configured
            # Handle this case: maybe append base_logits or a default value?
            # For now, let's log a warning and skip appending for this batch.
            print(f"Warning: 'aggregated_lora_logit' not found or is None in validation_step for stage {self.current_stage_index}. Skipping batch.")
            # Optionally append base_logits if available and desired:
            # base_logits = outputs.get('base_logits')
            # if base_logits is not None:
            #     self.validation_step_outputs_preds.append(base_logits.squeeze(1))
            # else:
            #     print("Warning: Both aggregated_lora_logit and base_logits are None in validation.")
            return # Skip appending preds/gts for this batch

        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs_preds or not self.validation_step_outputs_gts:
             print("Validation epoch end: No outputs collected.")
             # Log default values or skip logging if desired
             self.log_dict({'val_ap': 0.0, 'val_acc': 0.0, 'val_ap_epoch': 0.0}, on_epoch=True, logger=True)
             return

        try:
            all_preds_tensor = torch.cat(self.validation_step_outputs_preds, 0)
            all_gts_tensor = torch.cat(self.validation_step_outputs_gts, 0)

            all_preds = all_preds_tensor.to(torch.float32).sigmoid().flatten().cpu().numpy()
            all_gts = all_gts_tensor.to(torch.float32).cpu().numpy()

            # Ensure validate function exists and handles potential errors
            acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)

            # Log using Lightning's built-in logger, step is handled automatically
            self.log_dict({
                'val_acc': acc,
                'val_ap': ap, # Use this key for ModelCheckpoint if monitor='val_ap'
                'val_racc': r_acc,
                'val_facc': f_acc
            }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # Ensure the monitored key is logged for ModelCheckpoint
            self.log('val_ap_epoch', ap, on_epoch=True, logger=False, sync_dist=True) # Keep original key if needed

        except Exception as e:
            print(f"Error during validation epoch end: {e}")
            # Log default values or handle error appropriately
            self.log_dict({'val_ap': 0.0, 'val_acc': 0.0, 'val_ap_epoch': 0.0}, on_epoch=True, logger=True)

        finally:
            # Always clear lists
            self.validation_step_outputs_preds.clear()
            self.validation_step_outputs_gts.clear()


    def configure_optimizers(self):
        """ Configures the optimizer for the CURRENT training stage. """
        print(f"Trainer: Configuring optimizer for stage {self.current_stage_index}.")

        params_to_optimize = []
        if hasattr(self.model, 'get_params_for_current_stage'):
             params_to_optimize = self.model.get_params_for_current_stage()
        else:
             print("Warning: Model does not have 'get_params_for_current_stage' method. Cannot configure optimizer.")
             # Return an optimizer with no parameters to avoid crashing Lightning
             optimizer = optim.AdamW([], lr=self.opt.train.get('learning_rate', 1e-4))
             return {"optimizer": optimizer}


        if not params_to_optimize:
            print(f"Warning: No trainable parameters found for stage {self.current_stage_index}. Returning optimizer with no params.")
            optimizer = optim.AdamW([], lr=self.opt.train.get('learning_rate', 1e-4))
            return {"optimizer": optimizer} # No scheduler needed if no params

        print(f"Optimizing {len(params_to_optimize)} parameter groups for stage {self.current_stage_index}.")

        # Use optimizer and scheduler configuration from Hydra config (self.opt)
        try:
            # Instantiate optimizer using Hydra config, passing only the relevant params
            optimizer = hydra.utils.instantiate(self.opt.train.optimizer, params=params_to_optimize)
            print(f"Optimizer initialized: {optimizer.__class__.__name__}")
        except Exception as e:
            print(f"Error instantiating optimizer from config: {e}. Falling back to AdamW.")
            optimizer = optim.AdamW(params_to_optimize, lr=self.opt.train.get('learning_rate', 1e-4))


        # Instantiate scheduler using Hydra config
        try:
            scheduler_config = self.opt.train.get('scheduler', None) # Use get for safety
            # Ensure scheduler config is present and has _target_
            if scheduler_config and scheduler_config.get('_target_'):
                 # Ensure optimizer is passed correctly
                 scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)
                 print("Configured LR scheduler:", scheduler.__class__.__name__)
                 return {
                     "optimizer": optimizer,
                     "lr_scheduler": {
                         "scheduler": scheduler,
                         "interval": scheduler_config.get("interval", "epoch"), # Default interval: epoch
                         "frequency": scheduler_config.get("frequency", 1),   # Default frequency: 1
                         # Monitor metric if ReduceLROnPlateau or similar
                         "monitor": scheduler_config.get("monitor", "val_loss")
                     },
                 }
            else:
                 print("No LR scheduler configuration found or missing _target_. Proceeding without scheduler.")
                 return {"optimizer": optimizer}
        except Exception as e:
            print(f"Error instantiating scheduler from config: {e}. Proceeding without scheduler.")
            return {"optimizer": optimizer}

    def test_step(self, batch, batch_idx, dataloader_idx=0): # Add dataloader_idx
        x, y = batch
        outputs = self(x)
        logits = outputs['logits'] # Get logits from model output dict
        preds = logits.squeeze(1)
        # Initialize dict for dataloader_idx if not present
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

                # Ensure calculate_acc_auc_f1 exists and handles potential errors
                r_acc, f_acc, acc, auc, f1, ap = calculate_acc_auc_f1(all_gts % 2, all_preds, threshold=0.5) # Assuming threshold 0.5

                prefix = f"{dataloader_idx}/" # Prefix by dataloader index
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

        self.test_results = log_dict # Store results on the module instance
        # Clear outputs for next test run if needed
        self.test_step_outputs.clear()
        # This dictionary is automatically logged by Lightning if returned
        # return log_dict # No need to return, results stored in self.test_results used by train.py