"""Training stages module."""

from noe_train.training.reward import RewardConfig, compute_phi, compute_step_reward, compute_terminal_reward


def __getattr__(name):
    if name == "StageAConfig":
        from noe_train.training.stage_a import StageAConfig
        return StageAConfig
    if name == "train_role":
        from noe_train.training.stage_a import train_role
        return train_role
    if name == "StageBConfig":
        from noe_train.training.stage_b import StageBConfig
        return StageBConfig
    if name == "train_stage_b":
        from noe_train.training.stage_b import train_stage_b
        return train_stage_b
    if name == "StageCConfig":
        from noe_train.training.stage_c import StageCConfig
        return StageCConfig
    if name == "train_stage_c":
        from noe_train.training.stage_c import train_stage_c
        return train_stage_c
    raise AttributeError(f"module 'noe_train.training' has no attribute {name!r}")


__all__ = [
    "RewardConfig",
    "StageAConfig",
    "StageBConfig",
    "StageCConfig",
    "compute_phi",
    "compute_step_reward",
    "compute_terminal_reward",
    "train_role",
    "train_stage_b",
    "train_stage_c",
]
