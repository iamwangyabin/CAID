import lightning as L
from lightning.pytorch.callbacks import Callback
import torch # Import torch for float('inf')

class RankIncrementCallback(Callback):
    """
    Lightning Callback to increment the active LoRA rank at specified step intervals.
    """
    def __init__(self, increase_interval_steps, increment_amount=1):
        """
        Args:
            increase_interval_steps (int): The interval (in global steps) at which to increment the rank.
            increment_amount (int): The amount to increment the rank by each time.
        """
        super().__init__()
        self.increase_interval_steps = increase_interval_steps
        self.increment_amount = increment_amount
        if self.increase_interval_steps <= 0:
             print("RankIncrementCallback: increase_interval_steps must be positive. Disabling rank increment.")
             # Use float('inf') to effectively disable the check
             self.increase_interval_steps = float('inf')

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int):
        """ Checks global step and increments rank if interval is reached. """
        if self.increase_interval_steps == float('inf'):
            return

        effective_step = trainer.global_step + 1 # The step number we are about to start

        if effective_step > 0 and effective_step % self.increase_interval_steps == 0:
            # Access the underlying model assuming pl_module has it as self.model
            if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'increment_active_rank'):
                # Check if the model is the correct type, just to be safe
                if hasattr(pl_module.model, 'max_rank_potential'):
                    print(f"RankIncrementCallback: Global step {trainer.global_step} completed. Incrementing rank before step {effective_step}.")
                    pl_module.model.increment_active_rank(self.increment_amount)
                else:
                    print("RankIncrementCallback: Found model, but it doesn't seem to be the dynamic LoRA model.")
            else:
                print("RankIncrementCallback: Could not find pl_module.model or model.increment_active_rank method.")