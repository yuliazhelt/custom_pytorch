import numpy as np
from .base import Criterion
from .activations import LogSoftmax

from scipy.special import softmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        B, N = input.shape
        return 1. / (B * N) * ((target - input) ** 2).sum(axis=1).sum(axis=0)

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        B, N = input.shape
        return 2. / (B * N) * (input - target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        B, C = input.shape
        logsm = self.log_softmax.compute_output(input)
        mask = np.zeros((B, C))
        mask[np.arange(B), target.T] = 1
        return -1. / B * (logsm * mask).sum(axis=1).sum(axis=0)

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        B, C = input.shape
        mask = np.zeros((B, C))  
        mask[np.arange(B), target.T] = 1
        return 1. / B * (softmax(input, axis=1) - mask)
