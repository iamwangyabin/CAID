import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

class StepOffsetCallback(pl.Callback):
    """
    Callback to manually adjust the step counter for loggers (like WandbLogger)
    to ensure continuous logging across multiple Trainer.fit() calls.
    """
    def __init__(self, step_offset=0):
        super().__init__()
        self.step_offset = step_offset
        self._applied_offset = False # Flag to apply offset only once per fit call

    def on_train_start(self, trainer, pl_module):
        """Reset flag at the start of each fit call."""
        self._applied_offset = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        Before the first training batch of a `fit` call, adjust the logger's step.
        We access the logger's internal step counter (often `_step` or via `experiment.step`).
        This needs to be done carefully as it relies on internal logger implementation details.
        """
        if not self._applied_offset:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    # Access wandb run's step (might differ across wandb/lightning versions)
                    # Common ways: logger.experiment.step, logger._step
                    try:
                        # Try setting via the experiment object (wandb run)
                        logger.experiment.step = self.step_offset + trainer.global_step
                    except AttributeError:
                        # Fallback: try accessing internal _step if experiment.step fails
                        try:
                           logger._step = self.step_offset + trainer.global_step
                        except AttributeError:
                           # If neither works, log a warning or raise error
                           print(f"Warning: Could not set step offset for WandbLogger.")
                           pass 
            self._applied_offset = True # Ensure offset is applied only once

    # No state_dict/load_state_dict needed as this callback is stateless
