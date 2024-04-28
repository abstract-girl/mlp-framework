import numpy as np

from modules.activation import ReLU, Sigmoid, Softmax
from modules.nn import MLP
from modules.dataloader import DataLoader
from modules.datasets import IrisDataset, MnistDataset, DiabetesDataset
from modules.loss import CrossEntropy, MSE, MAE
from modules.optimizer import SGD, AdaGrad, Adam, GradientClipping, MomentumSGD
from modules.trainer import Trainer
from modules.transformer import OneHotEncodingTransformer, NormalizeTransformer
from modules.mode import Mode
from modules.early_stopping import EarlyStopping
from variants import *

if __name__ == '__main__':
    try:
        np.random.seed(42)  # Для воспроизводимости результатов

      #   mnist_adam_mlp(show_chart=True)
        iris_mlp_momentumsdg(show_chart=True)
      #   diabetes_mlp(show_chart=True)

    except KeyboardInterrupt:
        print('Обучение было прервано пользователем.')
