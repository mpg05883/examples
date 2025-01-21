import logging
import os
from datetime import datetime

import torch
from torch.distributed import all_gather, get_world_size
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datautils import MyTrainDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")


LOCAL_RANK = int(os.environ["LOCAL_RANK"])
GLOBAL_RANK = int(os.environ["RANK"])


def get_global_rank():
    return GLOBAL_RANK


def get_local_rank():
    return LOCAL_RANK


def is_main_process():
    return GLOBAL_RANK == 0


def get_timestamp():
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%m/%d/%Y %H:%M:%S")
    return formatted_timestamp


def _print_helper(
    message: str,
    show_gpu: bool = False,
    show_timestamp: bool = False,
    debug: bool = False,
    warning: bool = False,
    error: bool = False,
    critical: bool = False,
    epoch: int = None,
):
    if epoch is not None:
        message = f"Epoch {epoch + 1} - {message}"

    if show_gpu:
        message = f"GPU {get_global_rank()} | {message}"

    if show_timestamp:
        message = f"{get_timestamp()} | {message}"

    if critical:
        logging.critical(message)
        return
    elif error:
        logging.error(message)
        return
    elif warning:
        logging.warning(message)
        return
    elif debug:
        logging.debug(message)
        return
    else:
        logging.info(message)


def print_dist(
    message: str,
    to_all: bool = False,
    show_gpu: bool = False,
    show_timestamp: bool = False,
    debug: bool = False,
    warning: bool = False,
    error: bool = False,
    critical: bool = False,
    epoch: int = None,
):
    if to_all:
        _print_helper(
            message,
            True,  # Show GPU when printing to all processses
            show_timestamp,
            debug,
            warning,
            error,
            critical,
            epoch,
        )
        return

    if not is_main_process():
        return

    _print_helper(
        message,
        show_gpu,
        show_timestamp,
        debug,
        warning,
        error,
        critical,
        epoch,
    )


def load_objects(args):
    train_set = MyTrainDataset(2048)
    val_set = MyTrainDataset(512)
    test_set = MyTrainDataset(256)
    train_loader = prepare_dataloader(train_set, args.batch_size)
    val_loader = prepare_dataloader(val_set, args.batch_size)
    test_loader = prepare_dataloader(test_set, args.batch_size)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_loader, val_loader, test_loader, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def aggregate_tensors(tensor: torch.Tensor):
    world_size = get_world_size()
    tensor_list = [
        torch.zeros_like(tensor, device=LOCAL_RANK) for _ in range(world_size)
    ]
    all_gather(tensor_list, tensor)
    stacked_tensor = torch.stack(tensor_list)
    return stacked_tensor


def aggregate_metrics(total, num_samples):
    """
    Takes the average of a metric, then aggregates the averages computed by all
    processes.

    Args:
        total: Total accumulated metric
        num_samples: Number of samples
    """
    # Compute this process's average
    local_average = total / num_samples
    local_average_tensor = torch.tensor([local_average], device=LOCAL_RANK)

    # Aggregate average tensors across all processes
    stacked_tensor = aggregate_tensors(local_average_tensor)

    # Get aggregated average
    aggregated_average = torch.mean(stacked_tensor)

    return aggregated_average
