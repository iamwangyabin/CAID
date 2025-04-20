import lightning as L
from lightning.pytorch.callbacks import Callback
import torch # Import torch for float('inf')

class RankIncrementCallback(Callback):
    """Lightning Callback to increment the active LoRA rank at specified epoch intervals."""
    def __init__(self, increase_interval_epochs=1, increment_amount=1):
        """
        Args:
            increase_interval_epochs (int): The interval (in epochs) at which to increment the rank. Starts checking after the first epoch.
            increment_amount (int): The amount to increment the rank by each time.
        """
        super().__init__()
        self.increase_interval_epochs = increase_interval_epochs
        self.increment_amount = increment_amount
        if self.increase_interval_epochs <= 0:
             print("RankIncrementCallback: increase_interval_epochs must be positive. Disabling rank increment.")
             # Use float('inf') to effectively disable the check
             self.increase_interval_epochs = float('inf')

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """ Checks current epoch and increments rank if interval is reached. """
        if self.increase_interval_epochs == float('inf'):
            return

        # trainer.current_epoch starts at 0. We increment rank *after* an interval of epochs has passed.
        # So, if interval is 1, increment happens at the start of epoch 1, 2, 3...
        # If interval is 2, increment happens at the start of epoch 2, 4, 6...
        current_epoch = trainer.current_epoch

        if current_epoch > 0 and current_epoch % self.increase_interval_epochs == 0:
            # Access the underlying model assuming pl_module has it as self.model
            if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'increment_active_rank'):
                if hasattr(pl_module.model, 'max_rank_potential'):
                    print(f"RankIncrementCallback: Epoch {current_epoch-1} finished. Incrementing rank at start of epoch {current_epoch}.")
                    pl_module.model.increment_active_rank(self.increment_amount)
                else:
                    print("RankIncrementCallback: Found model, but it doesn't seem to be the dynamic LoRA model.")
            else:
                print("RankIncrementCallback: Could not find pl_module.model or model.increment_active_rank method.")