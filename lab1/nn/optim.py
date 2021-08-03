import nn.tensor
from .tensor import Tensor
from .modules import Module


class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):

        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.

        if isinstance(module, Module):
            for i in range(len(module.sequential)):
                for j in module.need_update:
                    self._update_weight(module.sequential[i][j].tensor, str(i)+str(j))
        elif isinstance(module, Tensor):
            self._update_weight(module)
        elif isinstance(module, list):
            for i in range(len(module)):
                self._step_module(module[i])

        # End of todo

    def _update_weight(self, tensor, pos):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum
        self.v = {}

    def _update_weight(self, tensor, pos):

        # TODO Update the weight of tensor
        # in SGD manner.

        if pos not in self.v.keys():
            self.v[pos] = nn.tensor.from_array(tensor.grad)
        else:
            self.v[pos] = self.v[pos]*self.momentum+tensor.grad
        tensor -= self.lr*self.v[pos]

        # End of todo


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.

        self.beta = [0.9, 0.999]
        self.figure = {}
        self.epsilon = 1e-8

        # End of todo

    def _update_weight(self, tensor, pos):

        # TODO Update the weight of
        # tensor in Adam manner.

        if pos not in self.figure.keys():
            self.figure[pos] = [1, nn.tensor.from_array(tensor.grad)*(1-self.beta[0]),
                                nn.tensor.from_array(tensor.grad**2)*(1-self.beta[1])]
        else:
            self.figure[pos][0] += 1
            self.figure[pos][1] = self.figure[pos][1] * self.beta[0] + tensor.grad * (1 - self.beta[0])
            self.figure[pos][2] = self.figure[pos][2] * self.beta[1] + (tensor.grad ** 2) * (1 - self.beta[1])
        f = self.figure[pos]
        lr = self.lr*(1-self.beta[1]**f[0])**0.5 / (1 - self.beta[0]**f[0])
        tensor -= lr * f[1] / (f[2]**0.5 + self.epsilon)

        # End of todo
