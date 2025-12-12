"""
Distributed training utilities for PyTorch.

This module provides helper functions for multi-GPU distributed training including:
- Distributed backend initialization
- Process synchronization
- Random seed management
- Master process identification
"""

import os
import random
import torch
import torch.distributed as dist
import numpy as np


def is_dist_avail_and_initialized() -> bool:
    """
    Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed backend is available and initialized, False otherwise.

    Note:
        This should be called before any distributed operations to avoid errors.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """
    Get the number of processes in the distributed group.

    Returns:
        int: Number of processes (GPUs) in the distributed training setup.
            Returns 1 if not in distributed mode.

    Example:
        >>> world_size = get_world_size()
        >>> print(f"Training on {world_size} GPUs")
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the rank of the current process.

    Returns:
        int: Rank of the current process (0 to world_size-1).
            Returns 0 if not in distributed mode.

    Note:
        Rank 0 is conventionally the master process which handles
        logging, checkpointing, and other I/O operations.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """
    Check if current process is the main (rank 0) process.

    Returns:
        bool: True if this is the main process, False otherwise.

    Usage:
        Use this to ensure only one process performs I/O operations:
        >>> if is_main_process():
        >>>     torch.save(model.state_dict(), 'checkpoint.pth')
        >>>     print("Model saved!")
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save model or checkpoint only on the master process.

    This is a wrapper around torch.save that ensures saving happens only
    on the master process to avoid conflicts and redundant saves.

    Args:
        *args: Positional arguments passed to torch.save
        **kwargs: Keyword arguments passed to torch.save

    Example:
        >>> save_on_master(model.state_dict(), 'checkpoint.pth')
        # Only rank 0 will actually save the file
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master: bool):
    """
    Setup print function for distributed training.

    This function modifies the built-in print to only output from the master
    process, preventing duplicate logging from all processes.

    Args:
        is_master (bool): Whether the current process is the master process.

    Technical Details:
        - Overrides the built-in print function
        - Non-master processes will only print if force=True is passed
        - Maintains all original print functionality

    Example:
        >>> setup_for_distributed(is_master=True)
        >>> print("This will print")  # Only from master process
        >>> print("Force print", force=True)  # Prints from all processes
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print_wrapper(*args, **kwargs):
        """
        Wrapper around print that respects distributed training setup.

        Args:
            *args: Arguments to print
            force (bool): If True, print even from non-master processes
            **kwargs: Other keyword arguments for print
        """
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print_wrapper


def init_distributed():
    """
    Initialize the distributed training backend.

    This function initializes PyTorch's distributed backend using environment
    variables set by torch.distributed.launch or torchrun.

    Environment Variables Required:
        RANK: Global rank of the current process
        WORLD_SIZE: Total number of processes
        LOCAL_RANK: Local rank on the current node

    Technical Details:
        - Uses NCCL backend for optimal GPU communication
        - Sets the CUDA device to LOCAL_RANK
        - Synchronizes all processes before proceeding
        - Sets up print suppression for non-master processes

    Usage:
        Launch with torch.distributed.launch:
        >>> # python -m torch.distributed.launch --nproc_per_node=4 train.py
        >>> init_distributed()
        >>> # Now you can use DDP and other distributed features

    Raises:
        KeyError: If required environment variables are not set
        RuntimeError: If CUDA device cannot be set

    Note:
        This function must be called before any model initialization or
        data loading when using distributed training.
    """
    # Get distributed parameters from environment
    dist_url = "env://"  # Use environment variables for configuration
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # NCCL is optimal for NVIDIA GPUs
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # Set CUDA device for this process
    try:
        torch.cuda.set_device(local_rank)
    except Exception as e:
        print(f"Error setting CUDA device {local_rank}: {e}", force=True)
        raise

    # Synchronize all processes
    dist.barrier()

    # Setup print suppression for non-master processes
    setup_for_distributed(rank == 0)


def seed_everything(seed: int = 1234):
    """
    Set random seeds for reproducibility across all libraries.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch CPU operations
    - PyTorch CUDA operations
    - CuDNN (deterministic mode)

    Args:
        seed (int): Random seed value. Default is 1234.

    Technical Details:
        - Sets torch.backends.cudnn.deterministic = True for reproducibility
        - This may reduce performance but ensures consistent results
        - Must be called before any random operations

    Example:
        >>> seed_everything(42)
        >>> # All random operations will be reproducible

    Note:
        Perfect reproducibility in distributed training may still be
        challenging due to non-deterministic GPU operations. For best
        results, also set PYTHONHASHSEED environment variable before
        starting Python:
        >>> # PYTHONHASHSEED=42 python train.py

    Warning:
        Setting cudnn.deterministic=True may reduce training speed.
        For faster training with less strict reproducibility, you can
        modify this function to use cudnn.benchmark=True instead.
    """
    # Python random seed
    random.seed(seed)

    # Set PYTHONHASHSEED for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary values across all processes in distributed training.

    This function synchronizes metrics (like loss, accuracy) across all GPUs
    by averaging or summing the values.

    Args:
        input_dict (dict): Dictionary with string keys and tensor values.
        average (bool): If True, average the values. If False, sum them. Default is True.

    Returns:
        dict: Dictionary with reduced values.

    Example:
        >>> metrics = {'loss': torch.tensor(0.5), 'acc': torch.tensor(0.9)}
        >>> avg_metrics = reduce_dict(metrics, average=True)
        >>> # avg_metrics contains average loss and accuracy across all GPUs

    Note:
        All values in the dictionary must be tensors. This function is useful
        for aggregating training metrics across multiple GPUs.
    """
    world_size = get_world_size()

    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        # Sort the keys for consistency across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        # Stack and reduce
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        # Create output dictionary
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict
