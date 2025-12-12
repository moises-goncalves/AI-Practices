"""
Configuration for 3D segmentation model training.

This config defines hyperparameters for training the ResNet18d-based
3D segmentation model used to generate organ masks.
"""

import sys
sys.path.append('./')
from paths import PATHS

try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except:
    device = 'cuda'

class CFG:
    """Configuration class for segmentation model training."""

    # Distributed training settings
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 3
    FOLD = 0

    # Model architecture
    model_name = 'resnet18d'
    V = '1'

    # Output directory
    OUTPUT_FOLDER = f"{PATHS.SEGMENTATION_MODEL_SAVE}/{model_name}_v{V}"

    # Random seed for reproducibility
    seed = 3407

    # Device configuration
    device = device
    
    n_folds = 5
    folds = [i for i in range(n_folds)]

    train_batch_size = 8
    valid_batch_size = 8
    acc_steps = 1
    
    lr = 5e-4
    wd = 1e-6
    n_epochs = 20
    n_warmup_steps = 0
    upscale_steps = 1.05
    validate_every = 1
    
    epoch = 0
    global_step = 0

    autocast = True

    workers = 2


