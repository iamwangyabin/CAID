# Import modules for registration side effects.
from .base import ContinualMethod
from .sequential_finetune import SequentialFineTune
from .cored import CoReDMethod
from .cddb import CDDBBenchmarkMethod
from .dfil import DFILMethod
from .e3 import E3Method
from .ca_adapter_cail import ContentAgnosticAdapterCAIL
from .hsic_bottleneck import HSICBottleneckMethod
from .saido import SAIDOMethod
from .prompt2guard import Prompt2GuardMethod
from .sprompts import SPromptsMethod
from .official_dil import (
    CPPromptMethod,
    DCEMethod,
    DUCTMethod,
    LayUPMethod,
    LoRanPACMethod,
    PINADMethod,
    PINAMethod,
    RanPACMethod,
    SOYOMethod,
)
from .hdp import HDPMethod
from .sur_lid import SURLIDMethod

__all__ = [
    "ContinualMethod",
    "SequentialFineTune",
    "CoReDMethod",
    "CDDBBenchmarkMethod",
    "DFILMethod",
    "E3Method",
    "ContentAgnosticAdapterCAIL",
    "HSICBottleneckMethod",
    "SAIDOMethod",
    "Prompt2GuardMethod",
    "SPromptsMethod",
    "RanPACMethod",
    "LayUPMethod",
    "PINAMethod",
    "PINADMethod",
    "CPPromptMethod",
    "DUCTMethod",
    "SOYOMethod",
    "LoRanPACMethod",
    "DCEMethod",
    "HDPMethod",
    "SURLIDMethod",
]
