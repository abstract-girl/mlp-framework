import unittest
from modules.optimizer import SGD, MomentumSGD, GradientClipping, Adam, AdaGrad
import numpy as np


class TestSGD(unittest.TestCase):
    def test_step(self):
        optimizer = SGD(lr=0.01)
        parameters = {'param1': [np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])],
                      'param2': [np.array([0.5, 1.5, 2.5]), np.array([0.2, 0.3, 0.4])]}

        updated_parameters = optimizer.step(parameters)

        # Исправлены индексы для проверки обновленных значений параметров, а не градиентов
        np.testing.assert_array_almost_equal(updated_parameters['param1'][0], np.array([0.999, 1.998, 2.997]),
                                             decimal=3)
        np.testing.assert_array_almost_equal(updated_parameters['param2'][0], np.array([0.498, 1.497, 2.496]),
                                             decimal=3)


class TestMomentumSGD(unittest.TestCase):
    def test_step(self):
        optimizer = MomentumSGD(lr=0.01, momentum=0.9)
        parameters = {'param1': [np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])],
                      'param2': [np.array([0.5, 1.5, 2.5]), np.array([0.2, 0.3, 0.4])]}

        # Выполнение первого шага
        optimizer.step(parameters)

        # Выполнение второго шага
        updated_parameters = optimizer.step(parameters)

        # Ожидаемые значения после второго шага
        expected_param1 = np.array([0.9971, 1.9942, 2.9913])
        expected_param2 = np.array([0.4942, 1.4913, 2.4884])

        np.testing.assert_array_almost_equal(updated_parameters['param1'][0], expected_param1, decimal=4)
        np.testing.assert_array_almost_equal(updated_parameters['param2'][0], expected_param2, decimal=4)


class TestGradientClippingForExecution(unittest.TestCase):
    def test_gradient_clipping_effect_on_parameters(self):
        optimizer = GradientClipping(lr=0.01, momentum=0.9, max_norm=1.0)
        parameters = {
            'param1': [np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5])]
        }

        # Выполнение шага оптимизации
        optimizer.step(parameters)

        # Ожидаем, что после обрезки градиентов и обновления параметров,
        # значения параметров изменятся меньше, чем без обрезки.
        # Проверяем фактическое изменение параметров
        expected_param1_after_clipping = np.array([0.995, 1.995, 2.995])
        np.testing.assert_array_almost_equal(parameters['param1'][0], expected_param1_after_clipping, decimal=3,
                                             err_msg="Parameters were not updated correctly after gradient clipping")


class TestAdam(unittest.TestCase):
    def test_step(self):
        optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        parameters = {
            'param1': [np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])]
        }

        # Выполнение одного шага оптимизации
        updated_parameters = optimizer.step(parameters)

        # Ожидаемые значения после первого шага оптимизации
        expected_param1 = np.array([0.999, 1.999, 2.999])

        # Проверка обновленных параметров
        np.testing.assert_array_almost_equal(updated_parameters['param1'][0], expected_param1, decimal=3,
                                             err_msg="Adam optimizer did not update parameters correctly")


class TestAdaGrad(unittest.TestCase):
    def test_step(self):
        optimizer = AdaGrad(lr=0.01, epsilon=1e-8)
        parameters = {
            'param1': [np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])]
        }

        # Выполнение одного шага оптимизации
        updated_parameters = optimizer.step(parameters)

        # Ожидаемые значения после первого шага оптимизации
        expected_param1 = np.array([0.99, 1.99, 2.99])

        # Проверка обновленных параметров
        np.testing.assert_array_almost_equal(updated_parameters['param1'][0], expected_param1, decimal=2,
                                             err_msg="AdaGrad optimizer did not update parameters correctly")


if __name__ == '__main__':
    unittest.main()
