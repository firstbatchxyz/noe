"""NoE expert implementations."""

from noe_train.experts.base import Chunk, ExpertOutput


def __getattr__(name):
    if name == "BaseExpert":
        from noe_train.experts.base import BaseExpert
        return BaseExpert
    if name == "CoderExpert":
        from noe_train.experts.coder import CoderExpert
        return CoderExpert
    if name == "DebuggerExpert":
        from noe_train.experts.debugger import DebuggerExpert
        return DebuggerExpert
    if name == "PlannerExpert":
        from noe_train.experts.planner import PlannerExpert
        return PlannerExpert
    if name == "TesterExpert":
        from noe_train.experts.tester import TesterExpert
        return TesterExpert
    if name == "get_lora_config":
        from noe_train.experts.lora_config import get_lora_config
        return get_lora_config
    raise AttributeError(f"module 'noe_train.experts' has no attribute {name!r}")


__all__ = [
    "BaseExpert",
    "Chunk",
    "CoderExpert",
    "DebuggerExpert",
    "ExpertOutput",
    "PlannerExpert",
    "TesterExpert",
    "get_lora_config",
]
