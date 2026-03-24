"""Router module."""


def __getattr__(name):
    if name == "RouterModel":
        from noe_train.router.model import RouterModel
        return RouterModel
    if name == "RouterOutput":
        from noe_train.router.model import RouterOutput
        return RouterOutput
    if name == "greedy_knapsack":
        from noe_train.router.packer import greedy_knapsack
        return greedy_knapsack
    if name == "StatePacket":
        from noe_train.router.state_packet import StatePacket
        return StatePacket
    if name == "compile_state":
        from noe_train.router.state_packet import compile_state
        return compile_state
    raise AttributeError(f"module 'noe_train.router' has no attribute {name!r}")


__all__ = ["RouterModel", "RouterOutput", "StatePacket", "compile_state", "greedy_knapsack"]
