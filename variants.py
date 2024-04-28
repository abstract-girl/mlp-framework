import numpy as np

from modules.activation import ReLU, Sigmoid, Softmax
from modules.nn import MLP
from modules.dataloader import DataLoader
from modules.datasets import IrisDataset, MnistDataset, DiabetesDataset
from modules.loss import CrossEntropy, MSE, MAE
from modules.optimizer import SGD, AdaGrad, Adam, GradientClipping, MomentumSGD
from modules.trainer import Trainer
from modules.transformer import OneHotEncodingTransformer, NormalizeTransformer, MinMaxScalerTransformer
from modules.mode import Mode
from modules.early_stopping import EarlyStopping


def iris_mlp_momentumsdg(show_chart=False):
    iris_ds = IrisDataset()

    input_size = iris_ds.features.shape[1]
    num_classes = len(iris_ds.labels)
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]
    mlp = MLP(input_size=input_size,
              mlp_hidden_config=[(4, ReLU)],
              num_classes=num_classes, last_activation=Softmax()
              )

    trainer = Trainer(model=mlp,
                      loss_fn=CrossEntropy(),
                      optimizer=MomentumSGD(lr=0.001),
                      dataset=iris_ds,
                      test_split_ratio=0.2, val_split_ratio=0.2, batch_size=10, shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=5000, show_chart=show_chart)

    params_model_fitted = trainer.model.parameters

    trainer.test()


def iris_mlp_sdg():
    iris_ds = IrisDataset()

    input_size = iris_ds.features.shape[1]
    num_classes = len(iris_ds.labels)
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]
    mlp = MLP(input_size=input_size,
              mlp_hidden_config=[(4, ReLU)],
              num_classes=num_classes, last_activation=Softmax()
              )

    trainer = Trainer(model=mlp,
                      loss_fn=CrossEntropy(),
                      optimizer=SGD(lr=0.001),
                      dataset=iris_ds,
                      test_split_ratio=0.2, val_split_ratio=0.2, batch_size=10, shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=10000, )

    params_model_fitted = trainer.model.parameters


def iris_mlp_gradientclipping():
    iris_ds = IrisDataset()

    input_size = iris_ds.features.shape[1]
    num_classes = len(iris_ds.labels)
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]
    mlp = MLP(input_size=input_size,
              mlp_hidden_config=[(4, ReLU)],
              num_classes=num_classes, last_activation=Softmax()
              )

    trainer = Trainer(model=mlp,
                      loss_fn=CrossEntropy(),
                      optimizer=GradientClipping(lr=0.001),
                      dataset=iris_ds,
                      test_split_ratio=0.2, val_split_ratio=0.2, batch_size=10, shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=10000, )

    params_model_fitted = trainer.model.parameters


def iris_mlp_adagrad():
    iris_ds = IrisDataset()

    input_size = iris_ds.features.shape[1]
    num_classes = len(iris_ds.labels)
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]
    mlp = MLP(input_size=input_size,
              mlp_hidden_config=[(4, ReLU)],
              num_classes=num_classes, last_activation=Softmax()
              )

    trainer = Trainer(model=mlp,
                      loss_fn=CrossEntropy(),
                      optimizer=AdaGrad(lr=0.01),
                      dataset=iris_ds,
                      test_split_ratio=0.2, val_split_ratio=0.2, batch_size=10, shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=10000, )

    params_model_fitted = trainer.model.parameters


def iris_mlp_adam():
    iris_ds = IrisDataset()

    input_size = iris_ds.features.shape[1]
    num_classes = len(iris_ds.labels)
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]
    mlp = MLP(input_size=input_size,
              mlp_hidden_config=[(4, ReLU)],
              num_classes=num_classes, last_activation=Softmax()
              )

    trainer = Trainer(model=mlp,
                      loss_fn=CrossEntropy(),
                      optimizer=Adam(lr=0.01),
                      dataset=iris_ds,
                      test_split_ratio=0.2, val_split_ratio=0.2, batch_size=10, shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=1000, )

    params_model_fitted = trainer.model.parameters


def mnist_adam_mlp(show_chart=False):
    ds = MnistDataset()

    input_size = ds.features.shape[1]
    num_classes = len(ds.labels)
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]
    mlp = MLP(input_size=input_size,
              mlp_hidden_config=[(128, ReLU), (64, ReLU)],
              num_classes=num_classes, last_activation=Softmax()
              )

    trainer = Trainer(model=mlp,
                      loss_fn=CrossEntropy(),
                      optimizer=Adam(lr=0.001),
                      dataset=ds,
                      test_split_ratio=0.2, val_split_ratio=0.2, batch_size=64, shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=20, early_stopping=EarlyStopping(patience=20), show_chart=show_chart)

    params_model_fitted = trainer.model.parameters

    trainer.test()


def diabetes_mlp(show_chart=False):
    ds = DiabetesDataset()

    input_size = ds.features.shape[1]

    normalize_transformer = NormalizeTransformer()

    num_classes = 1
    mlp = MLP(input_size=input_size,
              mlp_hidden_config=[(32, ReLU), (16, ReLU)],
              num_classes=num_classes,
              mode=Mode.REGRESSION
              )

    trainer = Trainer(model=mlp,
                      loss_fn=MSE(),
                      optimizer=Adam(lr=0.1),
                      dataset=ds,
                      test_split_ratio=0.2,
                      val_split_ratio=0.2,
                      batch_size=32,
                      shuffle=True,
                      mode=Mode.REGRESSION,
                      feature_transformers=[normalize_transformer]
                      )

    trainer.train(epochs=5000, validation_freq=10, show_chart=show_chart)
    params_model_fitted = trainer.model.parameters

    trainer.test()


def iris_classification_with_adam_and_early_stopping():
    dataset = IrisDataset()
    input_size = dataset.features.shape[1]
    num_classes = len(dataset.labels)

    model = MLP(input_size=input_size,
                mlp_hidden_config=[(10, ReLU), (10, ReLU)],
                num_classes=num_classes,
                last_activation=Softmax())

    optimizer = Adam(lr=0.001)
    loss_fn = CrossEntropy()
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      dataset=dataset,
                      test_split_ratio=0.2,
                      val_split_ratio=0.2,
                      batch_size=16,
                      shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=1000, early_stopping=EarlyStopping(patience=10))
    trainer.test()


def diabetes_regression_with_sgd_and_normalization():
    dataset = DiabetesDataset()
    input_size = dataset.features.shape[1]

    model = MLP(input_size=input_size,
                mlp_hidden_config=[(32, ReLU), (16, ReLU)],
                num_classes=1,
                mode=Mode.REGRESSION)

    optimizer = SGD(lr=0.01)
    loss_fn = MSE()
    feature_transformers = [NormalizeTransformer()]

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      dataset=dataset,
                      test_split_ratio=0.2,
                      val_split_ratio=0.2,
                      batch_size=32,
                      shuffle=True,
                      mode=Mode.REGRESSION,
                      feature_transformers=feature_transformers)

    trainer.train(epochs=1000, validation_freq=10)
    trainer.test()


def mnist_classification_with_adagrad():
    dataset = MnistDataset()
    input_size = dataset.features.shape[1]
    num_classes = len(dataset.labels)

    model = MLP(input_size=input_size,
                mlp_hidden_config=[(128, ReLU), (64, ReLU)],
                num_classes=num_classes,
                last_activation=Softmax())

    optimizer = AdaGrad(lr=0.005)
    loss_fn = CrossEntropy()
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      dataset=dataset,
                      test_split_ratio=0.2,
                      val_split_ratio=0.2,
                      batch_size=64,
                      shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=50, early_stopping=EarlyStopping(patience=5))
    trainer.test()


def iris_with_momentumsgd_and_gradient_clipping():
    dataset = IrisDataset()
    input_size = dataset.features.shape[1]
    num_classes = len(dataset.labels)

    model = MLP(input_size=input_size,
                mlp_hidden_config=[(8, ReLU)],
                num_classes=num_classes,
                last_activation=Softmax())

    optimizer = MomentumSGD(lr=0.01, momentum=0.9)
    loss_fn = CrossEntropy()
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      dataset=dataset,
                      test_split_ratio=0.2,
                      val_split_ratio=0.2,
                      batch_size=16,
                      shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=200, early_stopping=EarlyStopping(patience=10))
    trainer.test()


def diabetes_regression_with_adam_and_minmax_scaling():
    dataset = DiabetesDataset()
    input_size = dataset.features.shape[1]

    model = MLP(input_size=input_size,
                mlp_hidden_config=[(20, ReLU), (10, ReLU)],
                num_classes=1,
                mode=Mode.REGRESSION)

    optimizer = Adam(lr=0.005)
    loss_fn = MSE()
    feature_transformers = [MinMaxScalerTransformer()]

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      dataset=dataset,
                      test_split_ratio=0.2,
                      val_split_ratio=0.2,
                      batch_size=20,
                      shuffle=True,
                      mode=Mode.REGRESSION,
                      feature_transformers=feature_transformers)

    trainer.train(epochs=1000, validation_freq=20)
    trainer.test()


def mnist_classification_with_momentumsgd_and_early_stopping():
    dataset = MnistDataset()
    input_size = dataset.features.shape[1]
    num_classes = len(dataset.labels)

    model = MLP(input_size=input_size,
                mlp_hidden_config=[(100, ReLU), (50, ReLU)],
                num_classes=num_classes,
                last_activation=Softmax())

    optimizer = MomentumSGD(lr=0.01, momentum=0.9)
    loss_fn = CrossEntropy()
    target_transformers = [OneHotEncodingTransformer(num_classes=num_classes)]

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      dataset=dataset,
                      test_split_ratio=0.2,
                      val_split_ratio=0.2,
                      batch_size=128,
                      shuffle=True,
                      target_transformers=target_transformers)

    trainer.train(epochs=100, early_stopping=EarlyStopping(patience=5))
    trainer.test()
