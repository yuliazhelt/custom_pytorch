import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        
        if self.bias is None:
            self.output = np.dot(input, self.weight.T)
        else:
            self.output = np.dot(input, self.weight.T) + self.bias
        return self.output    
        
    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        return np.dot(grad_output, self.weight)

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += np.dot(grad_output.T, input)
        if self.grad_bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]
        
        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.input_mean_sqr = None
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None

        
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        
        if self.training == True:
            B = input.shape[0]
            self.mean = input.sum(axis=0) * 1. / B
            self.input_mean = input - self.mean
            self.input_mean_sq = self.input_mean ** 2
            self.var = self.input_mean_sq.sum(axis=0) * 1. / B
            self.sqrt_var = (self.var + self.eps) ** (1 / 2)
            self.inv_sqrt_var = 1. / self.sqrt_var
            self.output = (self.input_mean) * self.inv_sqrt_var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * B * 1. / (B - 1) * self.var
        
        else:
            self.output = (input - self.running_mean) / (self.running_var + self.eps) ** (1 / 2)
            
        if self.affine:
            self.output *= self.weight
            self.output += self.bias
        
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        B, N = grad_output.shape
        if self.training == True:
            grad_inv_sqrt_var = (grad_output * self.input_mean).sum(axis=0)
            grad_sqrt_var = grad_inv_sqrt_var * (-1.) * (self.inv_sqrt_var ** 2)
            grad_var = grad_sqrt_var * self.inv_sqrt_var / 2
            grad_input_mean_sq = np.tile((grad_var * 1. / B).reshape(1, N), (B, 1))
            grad_input_mean = grad_output * np.tile(self.inv_sqrt_var.reshape(1, N), (B, 1)) + grad_input_mean_sq * 2 * self.input_mean
            grad_mean = (-grad_input_mean).sum(axis=0)
            grad_input = grad_input_mean + np.tile((grad_mean * 1. / B).reshape(1, N), (B, 1))
            
        else:
            grad_input = grad_output / (self.running_var + self.eps) ** (1 / 2)
        
        if self.weight is not None:
                grad_input *= self.weight 
                
        return grad_input
        
        
    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.training == True:
            B = input.shape[0]
            mean = input.sum(axis=0) * 1. / B
            input_mean = input - mean
            input_mean_sq = input_mean ** 2
            var = input_mean_sq.sum(axis=0) * 1. / B
            sqrt_var = (var + self.eps) ** (1 / 2)
            inv_sqrt_var = 1. / sqrt_var
            norm_input = input_mean * inv_sqrt_var 
        else:
            norm_input = (input - self.running_mean) / (self.running_var + self.eps) ** (1 / 2) 
        
        if self.grad_weight is not None:
            self.grad_weight += (grad_output * norm_input).sum(axis=0)
        if self.grad_bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.array]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training == True:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)
            self.output = 1. / (1 - self.p) * self.mask * input
        else:
            self.output = input
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training == True:
            return 1. / (1 - self.p) * self.mask * grad_output
        else:
            return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        output = input
        for module in self.modules:
            output = module.compute_output(output)
        return output 

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """ 
        grad = grad_output
        for i in range(len(self.modules)):
            module = self.modules[len(self.modules) - i - 1]
            module_input = input
            if (i != len(self.modules) - 1):
                module_input = self.modules[len(self.modules) - i - 2].output
            module.update_grad_parameters(module_input, grad)    
            grad = module.compute_grad_input(module_input, grad) 
        return grad

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.array]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.array]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
