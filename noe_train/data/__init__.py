"""Data pipeline module."""

from noe_train.data.teacher_gen import build_teacher_prompt, format_as_training_sample, validate_plan


def __getattr__(name):
    _lazy = {
        "load_nemotron_swe": ("noe_train.data.nemotron_swe", "load_nemotron_swe"),
        "process_localization": ("noe_train.data.nemotron_swe", "process_localization"),
        "process_repair": ("noe_train.data.nemotron_swe", "process_repair"),
        "process_test_generation": ("noe_train.data.nemotron_swe", "process_test_generation"),
        "build_role_datasets": ("noe_train.data.role_dataset", "build_role_datasets"),
        "load_role_dataset": ("noe_train.data.role_dataset", "load_role_dataset"),
        "verify_counts": ("noe_train.data.role_dataset", "verify_counts"),
        "build_rl_task_pool": ("noe_train.data.rl_tasks", "build_rl_task_pool"),
        "filter_stage_b": ("noe_train.data.rl_tasks", "filter_stage_b"),
        "load_rl_swe": ("noe_train.data.rl_tasks", "load_rl_swe"),
        "load_swebench_train": ("noe_train.data.rl_tasks", "load_swebench_train"),
        "load_swebench_verified": ("noe_train.data.rl_tasks", "load_swebench_verified"),
    }
    if name in _lazy:
        mod_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'noe_train.data' has no attribute {name!r}")
