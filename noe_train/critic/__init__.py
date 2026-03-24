"""Critic module."""


def __getattr__(name):
    if name == "CriticModel":
        from noe_train.critic.model import CriticModel
        return CriticModel
    raise AttributeError(f"module 'noe_train.critic' has no attribute {name!r}")


__all__ = ["CriticModel"]
