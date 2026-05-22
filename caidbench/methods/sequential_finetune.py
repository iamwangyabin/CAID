from __future__ import annotations

from ..registry import register_method
from .base import ContinualMethod


@register_method("seq_ft")
@register_method("finetune")
class SequentialFineTune(ContinualMethod):
    """Plain sequential fine-tuning baseline."""
