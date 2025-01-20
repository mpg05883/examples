import logging
import os

import torch
from torch.distributed import all_gather, get_world_size, init_process_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datautils import MyTrainDataset

logging.basicConfig(level=logging.DEBUG)


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def is_main_process():
    return int(os.environ["RANK"]) == 0


def print_rank_0(
    message: str,
    debug: bool = False,
    warning: bool = False,
    critical: bool = False,
):
    if not is_main_process():
        return
    if critical:
        logging.critical(message)
        return
    if warning:
        logging.warning(message)
        return
    if debug:
        logging.debug(message)
        return
    logging.critical(message)


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
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = get_world_size()
    tensor_list = [
        torch.zeros_like(tensor, device=local_rank) for _ in range(world_size)
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
    local_rank = int(os.environ["LOCAL_RANK"])
    local_average_tensor = torch.tensor(
        local_average,
        dtype=torch.float32,
        device=local_rank,
    )

    # Aggregate average tensors across all processes
    stacked_tensor = aggregate_tensors(local_average_tensor)

    # Get aggregated average
    aggregated_average = torch.mean(stacked_tensor)

    return aggregated_average
