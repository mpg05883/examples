import os
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

from util import aggregate_metrics, aggregate_tensors, is_main_process, print_rank_0


class Trainer:
    def __init__(
        self,
        args: Namespace,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        snapshots_directory: str = "snapshots",
        from_scratch: bool = True,
    ):
        self.args = args
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.epochs_ran = 0
        self.snapshots_directory = snapshots_directory
        if not os.path.exists(self.snapshots_directory):
            os.makedirs(self.snapshots_directory, exist_ok=True)
        self.snapshot_file_path = os.path.join(
            self.snapshots_directory,
            self.args.snapshot_file_name,
        )
        if not from_scratch:
            self._load_snapshot()

    def _load_snapshot(self):
        if not os.path.exists(self.snapshot_file_path):
            print_rank_0(
                "Cannot find previous snapshot. Starting training from epoch 0",
                warning=True,
            )
            return

        map_location = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.snapshot_file_path, map_location=map_location)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_ran = snapshot["EPOCHS_RAN"]
        print_rank_0(f"Resuming training from Epoch {self.epochs_ran + 1}")

    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RAN": epoch,
        }
        torch.save(snapshot, self.snapshot_file_path)
        print_rank_0(f"Snapshot saved to {self.snapshot_file_path}...\n")

    def _tensors_to_numpy(self, outputs, targets):
        outputs = outputs.view(-1).cpu().numpy()
        targets = targets.view(-1).cpu().numpy()
        return outputs, targets

    def _read_columns_from_csv(self, file_name: str = "values.csv", directory="data"):
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        return tuple(df[col] for col in df.columns)

    def _write_tensors_to_csv(
        self,
        outputs,
        targets,
        file_name: str = "values.csv",
        directory="data",
    ):
        outputs, targets = self._tensors_to_numpy(outputs, targets)
        data = {
            "pred": outputs,
            "true": targets,
        }
        df = pd.DataFrame(data)

        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        df.to_csv(file_path, index=False)

    def _create_plot(
        self,
        outputs,
        targets,
        file_name="my_plot.png",
        directory="plots",
    ):
        # If outputs and targets are tensors, convert them into numpy arrays
        if torch.is_tensor(outputs) and torch.is_tensor(targets):
            outputs, targets = self._tensors_to_numpy(outputs, targets)

        # Create plot
        plt.figure()
        plt.plot(targets, label="True", linewidth=2)
        plt.plot(outputs, label="Pred", linewidth=2)
        plt.title("True vs Predicted Values")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

        # Save plot
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        plt.savefig(file_path, bbox_inches="tight")

    def _read_and_plot(self):
        outputs, targets = self._read_columns_from_csv()
        self._create_plot(outputs, targets)

    def _write_and_plot(self, outputs, targets):
        self._write_tensors_to_csv(outputs, targets)
        self._create_plot(outputs, targets)

    def _forward_pass(self, source):
        # Clear gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(source)
        return outputs

    def _run_batch(self, source, targets):
        outputs = self._forward_pass(source)

        loss = F.mse_loss(outputs, targets)

        batch_size = source.size(0)

        return loss, batch_size

    def evaluate(self, epoch: int = 0, test: bool = True):
        if is_main_process():
            self._read_and_plot()

        mean_absolute_error = MeanAbsoluteError().to(self.local_rank)
        mean_squared_error = MeanSquaredError().to(self.local_rank)

        total_mae = 0.0  # Total MAE for a single process
        total_mse = 0.0  # Total MSE for a single process
        num_samples = 0  # Number of samples seen on a single process

        y_pred = torch.tensor([], device=self.local_rank)
        y_true = torch.tensor([], device=self.local_rank)

        # Set to evaluation mode
        self.model.eval()
        print_rank_0(f"Evaluation mode: {not self.model.module.training}", debug=True)

        with torch.no_grad():
            # Get correct dataloader
            dataloader = self.test_loader if test else self.val_loader
            dataloader.sampler.set_epoch(epoch)

            # Only show progress bar on main process
            if is_main_process():
                print("\n")
                pbar = tqdm(total=len(dataloader))

            for source, targets in dataloader:
                # Move tensors to GPU
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
                outputs = self._forward_pass(source)

                y_pred = torch.cat((y_pred, outputs))
                y_true = torch.cat((y_true, targets))

                batch_size = source.size(0)
                batch_mae = mean_absolute_error(targets, outputs)
                batch_mse = mean_squared_error(targets, outputs)
                total_mae += batch_mae * batch_size
                total_mse += batch_mse * batch_size
                num_samples += batch_size

                if is_main_process():
                    pbar.update(1)

        # Aggregate y_pred and y_true from all processes
        aggregated_outputs = aggregate_tensors(y_pred)
        aggregated_targets = aggregate_tensors(y_true)

        if is_main_process():
            pbar.close()
            self._write_and_plot(aggregated_outputs, aggregated_targets)

        # Aggregate MAE and MSE across all processes
        aggregated_mae = aggregate_metrics(total_mae, num_samples)
        aggregated_mse = aggregate_metrics(total_mse, num_samples)

        return aggregated_mae, aggregated_mse

    def train(self):
        # Set to training mode
        self.model.train()
        print_rank_0(f"Training mode: {self.model.module.training}", debug=True)

        for epoch in range(self.epochs_ran, self.args.num_epochs):
            # Total training loss on a single process
            total_train_loss = 0.0

            # Number of samples seen on a single process
            num_samples = 0

            # Set epoch
            self.train_loader.sampler.set_epoch(epoch)

            for source, targets in self.train_loader:
                # Move tensors to GPU
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
                loss, batch_size = self._run_batch(source, targets)

                # Backwards pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                total_train_loss += loss.item()
                num_samples += batch_size

            # Outside of for loop
            average_train_loss = aggregate_metrics(total_train_loss, num_samples)
            print_rank_0(f"Epoch {epoch + 1} | Train loss: {average_train_loss:.3e}")

            if epoch % self.args.save_every == 0:
                mae, mse = self.evaluate(epoch)
                print_rank_0(
                    f"Epoch {epoch + 1} | Validation MAE: {mae:.3f}, Validation MSE: {mse:.3f}"
                )
                if is_main_process():
                    self._save_snapshot(epoch)
