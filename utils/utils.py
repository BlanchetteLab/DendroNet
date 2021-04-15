import os
import shutil


def generate_default_config():
    return {
        'num_classes': 2,
        'lr': 0.001,
        'seed': 0,
        'l2': False,
        'delta_penalty_factor': 0.01,
        'l2_penalty_factor': 0.01,
        'num_steps': 5000,
        'validation_interval': 100,
        'loss_scale': 1.0
    }


def create_directory(path, remove_curr):
    if os.path.exists(path):
        if remove_curr:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
