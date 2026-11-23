"""Load the selected trainer without importing unrelated backbone code."""

from importlib import import_module


def __getattr__(name):
    if name in {'BaseTrainer', 'DINOv2TCNTrainer', 'RegionalDINOTrainer',
                'SSMambaTrainer', 'PhysnetTrainer', 'TscanTrainer',
                'EfficientPhysTrainer', 'DeepPhysTrainer', 'BigSmallTrainer',
                'PhysFormerTrainer'}:
        module = import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
