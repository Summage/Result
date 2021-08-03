import numpy as np

import nn.tensor
from .modules import Module


class Affine(Module):
    def __init__(self, tensor):
        super().__init__()
        if not isinstance(tensor, nn.tensor.Tensor):
            raise TypeError
        self.tensor = tensor
        self.grad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.tensor[1:, :]) + self.tensor[0]

    def backward(self, dy):
        delta = np.dot(dy, self.tensor[1:, :].T)
        self.tensor.grad += np.vstack([np.sum(dy, axis=0), np.dot(self.x.T, dy)])
        return delta


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # [1000, 1, 13, 13] -> [1000, 169]
        self.shape = x.shape
        return np.squeeze(x.reshape(*x.shape[:-2], -1))

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy.reshape(self.shape)


class Sigmoid(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of sigmoid function.

        self.x = None
        self.value = np.array(1.0 / (1.0 + np.exp(-x)))
        return self.value

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of sigmoid function.

        self.grad = dy * self.value * (1 - self.value)
        return self.grad

        # End of todo


class Tanh(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of tanh function.

        self.x = np.exp(x)
        self.value = (self.x - 1.0 / self.x) / (self.x + 1.0 / self.x)
        return self.value

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of tanh function.

        return 4.0 / ((self.x + 1.0 / self.x) ** 2)

        # End of todo


class ReLU(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of ReLU function.

        self.x = (x <= 0)
        self.value = x
        self.value[self.x] = 0
        return self.value

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of ReLU function.

        dy[self.x] = 0
        self.grad = dy
        return self.grad

        # End of todo


class Softmax(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of ReLU function.

        self.x = x
        self.exp_x = np.exp(x)
        self.value = np.sum(self.exp_x, axis=-2)
        self.value = np.array([v / s for v, s in zip(self.exp_x, self.value)])
        return self.value

        # End of todo

    def backward(self, dy=None):
        # TODO Implement backward propogation
        # of ReLU function.

        ...

        # End of todo


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):
        # TODO Calculate softmax loss.

        self.targets = targets
        self.probs = np.exp(probs)
        self.probs = self.probs / sum(self.probs)
        if self.probs.shape == self.targets.shape:
            self.targets = self.targets.argmax(axis=1)
        return -np.sum(np.log(self.probs[np.arange(self.probs.shape[0]), self.targets] + 1e-8)) / \
               self.probs.shape[0]

        # End of todo

    def backward(self):
        # TODO Implement backward propogation
        # of softmax loss function.

        tmp = np.copy(self.probs)
        tmp[np.arange(self.probs.shape[0]), self.targets] -= 1
        return tmp / self.probs.shape[0]

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.
        if probs.ndim == 1:
            probs = probs.reshape(1, probs.size)
            targets = targets.reshape(1, targets.size)
        # if targets.size != probs.size:
        #     self.targets = np.array([np.pad([1], (t, 9-t), constant_values=0) for t in targets] )
        # else:
        #     self.targets = targets
        self.probs = probs
        self.targets = targets
        if self.probs.size == self.targets.size:
            self.value = -sum(self.targets * np.log(self.probs + 1e-8)) / self.targets.shape[0]
        else:
            self.value = -sum(np.log(self.probs[np.arange(self.targets.shape[0]), self.targets] + 1e-8)) / \
                         self.targets.shape[0]
        # self.value = -sum(self.targets * np.log(self.probs + 1e-8)) / self.targets.shape[0]
        return self

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.

        if self.probs.shape == self.targets.shape:
            return (self.probs - self.targets) / self.probs.shape[0]
        else:
            tmp = np.copy(self.probs)
            tmp[np.arange(self.probs.shape[0]), self.targets] -= 1
            return tmp / self.probs.shape[0]

        # End of todo
