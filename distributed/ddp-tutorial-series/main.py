"""
Youtube video:
https://www.youtube.com/watch?v=KaAJtI1T2x4

PyTorch tutorial page:
https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html?utm_source=youtube&utm_medium=organic_social&utm_campaign=tutorial

GitHub repo:
https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblZjQ3U3WmI4VjFpV2k2Zi1hWlZvR0ZDcERRQXxBQ3Jtc0tuSnBucEdqTERIU3f2UWdrelUtSGk2eFYxelgwcm5neFAzaDNOcko3SkthRkRIelhBNUdEQ29ycFZ0LUNOZUxBNG5hY2d5SEhQb01MNDMxQ0tHaDczektlcS1hX0hDWDNxcmZCMk1iLWpDR0xSSXZ4TQ&q=https%3A%3f%3fgithub.com%3fpytorch%3fexamples%3fblob%3fmain%3fdistributed%3fddp-tutorial-series%3fmultinode.py&v=KaAJtI1T2x4

Medium article:
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.distributed import destroy_process_group, get_world_size, init_process_group

from trainer import Trainer
from util import get_global_rank, load_objects, print_dist


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def main(args):
    if not torch.cuda.is_available():
        print_dist(
            "ERROR: could not find GPU. At least one GPU is needed to run this"
            "script.\nEnding now...",
            critical=True,
        )
        return

    # Initialize process group
    ddp_setup()

    num_gpus = torch.cuda.device_count()
    print_dist(f"Number of GPUs avaliable: {num_gpus}", debug=True)

    world_size = get_world_size()
    print_dist(f"World size: {world_size}", debug=True)

    # Number of processes per node
    nproc_per_node = world_size // num_gpus
    print_dist(f"Number of processes per node: {nproc_per_node}", debug=True)

    num_nodes = world_size // nproc_per_node
    print_dist(f"Number of nodes: {num_nodes}", debug=True)

    global_rank = get_global_rank()
    print_dist(f"I'm GPU {global_rank}!", debug=True)

    # Load dataloaders, model, and optimizer
    train_loader, val_loader, test_loader, model, optimizer = load_objects(args)

    # Initialize trainer
    trainer = Trainer(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
    )

    start_time = time.time()

    # Train model
    trainer.train()

    # Test model
    mae, mse = trainer.evaluate(test=True)
    print_dist(f"Test MAE: {mae:.3f}, Test MSE: {mse:.3f}")

    time_elapsed_seconds = np.abs(time.time() - start_time)
    print_dist(f"Finished! Time elapsed: {time_elapsed_seconds:.3f} seconds\n")

    # Clean up process group
    destroy_process_group()


"""
How to run script:
torchrun --standalone --nproc_per_node=gpu main.py 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample distributed training and evaluation script"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Total number of epochs to train model",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="How frequently a snapshot will be saved in number of epochs",
    )
    parser.add_argument(
        "--snapshot_file_name",
        type=str,
        default="snapshot.pt",
        help="Name of file where snapshots will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size on each device",
    )
    args = parser.parse_args()
    main(args)
