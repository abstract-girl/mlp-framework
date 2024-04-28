from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from modules.nn import MLP
from modules.dataloader import DataLoader
from modules.loss import Loss
from modules.optimizer import Optimizer
from modules.datasets import Dataset
from modules.transformer import Transformer
from modules.mode import Mode
from modules.early_stopping import EarlyStopping
import matplotlib.pyplot as plt

from tqdm import tqdm


class Result:
    def __init__(self, loss, metric):
        self.loss = loss
        self.metric = metric


class Trainer:
    def __init__(self, model: MLP,
                 loss_fn: Loss,
                 optimizer: Optimizer,
                 dataset: Dataset,
                 test_split_ratio: float,
                 val_split_ratio: float,
                 batch_size: int,
                 shuffle: bool,
                 mode: Mode = Mode.CLASSIFICATION,
                 feature_transformers: List[Transformer] = None,
                 target_transformers: List[Transformer] = None):
        assert 0 <= test_split_ratio < 1, "Test split ratio must be between 0 and 1"
        assert 0 <= val_split_ratio < 1, "Validation split ratio must be between 0 and 1"
        assert test_split_ratio + val_split_ratio < 1, "Sum of test and validation split ratios must be less than 1"

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataset = dataset
        self.mode = mode

        self.settings_info = f"{self.model.__class__.__name__}, {self.loss_fn.__class__.__name__},{self.optimizer.__class__.__name__},{self.dataset.__class__.__name__}"

        self.train_losses = []
        self.val_losses = []

        # Splitting the dataset into training+validation and test datasets
        train_val_dataset, self.test_dataset = train_test_split(dataset, test_size=test_split_ratio, shuffle=shuffle)

        # Further splitting the training+validation dataset into training and validation datasets
        self.train_dataset, self.val_dataset = train_test_split(train_val_dataset,
                                                                test_size=val_split_ratio / (1 - test_split_ratio),
                                                                shuffle=shuffle)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle,
                                           feature_transformers=feature_transformers,
                                           target_transformers=target_transformers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle,
                                         feature_transformers=feature_transformers,
                                         target_transformers=target_transformers)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle,
                                          feature_transformers=feature_transformers,
                                          target_transformers=target_transformers)

    def train(self, epochs=10, validation_freq=1, early_stopping: EarlyStopping = None, show_chart=False):

        self.train_losses = []
        self.val_losses = []

        if show_chart:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))

        with tqdm(total=epochs, desc="Overall Progress", unit="epoch", dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                # Обучение
                result = self.evaluate(self.train_dataloader, training=True, pbar=pbar, epoch=epoch, phase="Train",
                                       show_progress=True)

                self.train_losses.append(result.loss)

                # Валидация с заданной частотой
                if (epoch + 1) % validation_freq == 0:
                    val_result = self.evaluate(self.val_dataloader, pbar=pbar, epoch=epoch, phase="Validation",
                                               show_progress=True)
                    self.val_losses.append(val_result.loss)

                    if early_stopping is not None:
                        early_stopping(val_result.loss)
                        if early_stopping.early_stop:
                            print("Stopped at epoch:", epoch + 1)
                            break
                else:
                    self.val_losses.append(None)

                pbar.update(1)

                if show_chart:
                    self.update_plot(epoch=epoch)

        if show_chart:
            plt.ioff()
            plt.show()

    def r2_score(self, y_true, y_pred):
        """
        Вычисляет коэффициент детерминации R^2.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    def show_result(self, name, result):
        if self.mode == Mode.CLASSIFICATION:
            print(f"{name} - Loss: {result.loss:.4f}, Accuracy: {result.metric:.4f}")
        elif self.mode == Mode.REGRESSION:
            print(f"{name} - Loss: {result.loss:.4f}, R^2: {result.metric:.4f}")

    def evaluate(self, dataloader, training=False, show_progress=False, pbar=None, epoch=None, phase="") -> Result:
        total_loss = 0
        if self.mode == Mode.CLASSIFICATION:
            correct_predictions = 0
            total_predictions = 0
        elif self.mode == Mode.REGRESSION:
            preds = []
            targets = []

        iterator = tqdm(dataloader, desc=f"{phase} Evaluating", leave=False,
                        dynamic_ncols=True) if show_progress else dataloader
        for batch in iterator:
            features, batch_targets = batch["features"], batch["targets"]
            predictions = self.model(features)
            loss = self.loss_fn.forward(predictions, batch_targets)
            total_loss += loss.item()

            if training:
                # Обратное распространение и оптимизация
                self.model.zero_grad()
                grad_loss = self.loss_fn.backward(predictions, batch_targets)
                self.model.backward(grad_loss)
                self.optimizer.step(parameters=self.model.parameters)

            if self.mode == Mode.CLASSIFICATION:
                predicted_labels = np.argmax(predictions, axis=1)
                targets_idx = np.argmax(batch_targets, axis=1)
                correct_predictions += (predicted_labels == targets_idx).sum()
                total_predictions += targets_idx.size
            elif self.mode == Mode.REGRESSION:
                preds.extend(predictions.flatten())
                targets.extend(batch_targets.flatten())

        average_loss = total_loss / len(dataloader)

        if self.mode == Mode.CLASSIFICATION:
            metric = correct_predictions / total_predictions
        elif self.mode == Mode.REGRESSION:
            metric = self.r2_score(np.array(targets), np.array(preds))

        if pbar is not None and epoch is not None:
            pbar.set_postfix(
                {"Epoch": epoch + 1, "Phase": phase, "Loss": f"{average_loss:.4f}", "Metric": f"{metric:.4f}"},
                refresh=True)

        return Result(average_loss, metric)

    def test(self):
        result = self.evaluate(self.test_dataloader)
        self.show_result("Test", result)

    def update_plot(self, epoch):
        self.ax.clear()
        # Перерисовываем график с учётом, что в val_losses могут быть None
        self.ax.plot(self.train_losses, label='Training Loss')
        # Фильтруем None и рисуем только те эпохи, для которых есть данные валидационного лосса
        val_losses_filtered = [loss for loss in self.val_losses if loss is not None]
        val_epochs = [epoch for epoch, loss in enumerate(self.val_losses) if loss is not None]
        if val_losses_filtered:
            self.ax.plot(val_epochs, val_losses_filtered, label='Validation Loss')
        self.ax.set_title(f'Loss over Epochs ({self.settings_info})')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.ax.grid(True)
        plt.draw()
        plt.pause(0.001)
