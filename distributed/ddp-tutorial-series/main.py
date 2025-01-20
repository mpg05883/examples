"""
Youtube video:
https://www.youtube.com/watch?v=KaAJtI1T2x4

PyTorch tutorial page:
https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html?utm_source=youtube&utm_medium=organic_social&utm_campaign=tutorial

GitHub repo:
https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblZjQ3U3WmI4VjFpV2k2Zi1hWlZvR0ZDcERRQXxBQ3Jtc0tuSnBucEdqTERIU3F2UWdrelUtSGk2eFYxelgwcm5neFAzaDNOcko3SkthRkRIelhBNUdEQ29ycFZ0LUNOZUxBNG5hY2d5SEhQb01MNDMxQ0tHaDczektlcS1hX0hDWDNxcmZCMk1iLWpDR0xSSXZ4TQ&q=https%3A%2F%2Fgithub.com%2Fpytorch%2Fexamples%2Fblob%2Fmain%2Fdistributed%2Fddp-tutorial-series%2Fmultinode.py&v=KaAJtI1T2x4

Medium article:
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
"""

import argparse

from torch.distributed import destroy_process_group

from trainer import Trainer
from util import ddp_setup, load_objects


def main(args):
    # Initialize process group
    ddp_setup()

    # Load dataloaders, model, and optimizer
    train_loader, val_loader, test_loader, model, optimizer = load_objects()

    # Initialize trainer
    trainer = Trainer(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
    )

    # Train model
    trainer.train()

    # Test model
    trainer.evaluate(val=False)

    # Clean up process group
    destroy_process_group()


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
