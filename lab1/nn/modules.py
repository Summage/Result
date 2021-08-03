import numpy as np
from itertools import product

import nn.tensor
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """

    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # TODO Initialize the weight
        # of linear module.

        self.w = tensor.from_array(np.random.normal(0, 1, [in_length + 1, out_length]))
        self.input = None

        # End of todo

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.

        if x.shape[1] != np.shape(self.w)[0] - 1:
            raise ValueError("Mismatch occurred in each input`s length!")
        self.input = x
        return np.dot(self.input, self.w[1:, :]) + self.w[0, :]

        # End of todo

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.

        if dy.shape[1] != self.w.shape[1]:
            raise ValueError("Mismatch occurred in each input`s length!")
        m = np.vstack([sum(dy), np.array(np.dot(self.input.T, dy))])
        self.w.grad = m

        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float = 0.1):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.

        # self.gamma = 1  # 权重
        # self.beta = 0  # 偏置
        self.tensor = nn.tensor.from_array(np.array([0., 1.]))  # [beta, gamma]
        self.epsilon = 1e-8
        self.momentum = momentum
        self.mu = 0.  # 均值
        self.sigma = 0.  # 标准差
        self.x_mu = None

        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.

        self.mu = self.momentum * self.mu + np.einsum('ij->j', x, optimize=True) / x.shape[0]
        self.x_mu = x - self.mu
        self.sigma = self.sigma * self.momentum + \
                     np.sqrt(np.einsum('ij,ij->j', self.x_mu, self.x_mu, optimize=True) /
                             x.shape[0] + self.epsilon)
        return self.tensor[1] * self.x_mu / self.sigma + self.tensor[0]

        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.

        self.tensor.grad = np.array([np.einsum('ij->', dy) / dy.size,
                                     np.einsum('ij->', dy * self.x_mu / self.sigma)])
        return dy / self.sigma
        # m = self.tensor.grad
        # dxhat = dy * self.tensor[1]
        # divar = np.sum(dxhat*self.x_mu, axis=0)
        # dxmu1 = dxhat * divar
        # dsqrtvar = -1./self.sigma*divar
        # dvar = 0.5 * 1. /self.sigma*dsqrtvar
        # dsq = 1./dy.shape[0]*np.ones(dy.shape)*dvar
        # dxmu2 = 2 * self.x_mu * dsq
        # dx1 = dxmu1 + dxmu2
        # dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        # dx2 = 1. / dy.shape[0] * np.ones(dy.shape) * dmu
        # dx = dx1 + dx2
        # return dx
        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.

        self.in_channels = in_channels
        self.out_channels = channels
        self.kernel_size = kernel_size
        self.kernel = nn.tensor.from_array(np.random.randn(self.in_channels, self.out_channels,
                                                           kernel_size, kernel_size))
        self.kernel.grad = np.zeros_like(self.kernel)
        self.k_col = None
        self.stride = stride
        self.padding = padding
        self.bias = None  # to be continued
        self.x = None
        self.x_padded = None
        self.col = None
        self.z = None
        self.tensor = self.kernel

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.

        k, s, p = self.kernel_size, self.stride, self.padding
        B, C_in, H_in, W_in = x.shape
        assert self.in_channels == C_in, '通道数不匹配'
        if s != 1:
            assert (H_in + 2 * p) % s == 0, '步长与高度不匹配'
            assert (W_in + 2 * p) % s == 0, '步长与宽度不匹配'
        self.x = x
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
        # col [B, C, H_out, W_out, H_in, W_in]
        col = Conv2d_im2col.forward(self, x_padded)
        self.col = col
        # 都拉伸的常规实现
        self.k_col = self.kernel.reshape(self.kernel.shape[0],-1).T
        self.z = np.dot(self.col, self.k_col).reshape(
            B, (H_in+p*2-k)//s+1, (W_in+2*p-k)//s+1, -1)
        self.z = np.transpose(self.z, (0,3,1,2))
        # 爱因斯坦和实现
        # self.z = np.einsum('bcklhw,cohw->bokl', col, self.kernel)
        # tensordot张量积实现
        # self.z = np.swapaxes(np.tensordot(col, self.kernel, axes=([1,4,5],[0,2,3])), -1, -3)

        if self.bias is True:
            ...
        # 无im2col优化的数学卷积
        # for n in np.arange(B):
        #     for c_out in np.arange(C_out):
        #         for h in np.arange(H_out):
        #             for w in np.arange(W_out):
        #                 self.z[n, c_out, h, w] = \
        #                     np.sum(x_padded[n, :, h * s: (h + k) * s, w * s: (w + k) * s])  # + self.bias[c_out]
        return self.z

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.

        # 对于步长大于1的降采样卷积，需要在dy中填充0
        # cache = [self.kernel, self.padding, self.kernel_size]
        # if self.stride > 1:
        #     for h in np.arange(dy.shape[-2] - 1, 0, -1):
        #         for _ in np.arange(self.stride - 1):
        #             dy = np.insert(dy, h, 0, axis=-2)
        #     for w in np.arange(dy.shape[-1] - 1, 0, -1):
        #         for _ in np.arange(self.stride - 1):
        #             dy = np.insert(dy, w, 0, axis=-1)
        #
        # kernel_t = np.swapaxes(np.flip(self.kernel, (2, 3)), 0, 1)
        # dy_padded = np.pad(dy, ((0, 0), (0, 0),
        #                         (self.kernel_size - 1, self.kernel_size - 1),
        #                         (self.kernel_size - 1, self.kernel_size - 1)),
        #                    'constant', constant_values=0)
        # self.kernel, self.padding = kernel_t, 0
        # dx = self.forward(dy_padded)
        #
        # x_t = np.swapaxes(self.x, 2, 3)
        # self.kernel, self.kernel_size = dy, dy.shape[-1]
        # dk = self.forward(x_t)/self.x.shape[0]
        #
        # self.kernel, self.padding, self.kernel_size = cache
        # self.kernel.grad += dk
        # if self.padding != 0:
        #     dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        # return dx

        dy = dy.transpose(0,2,3,1).reshape(-1, self.kernel.shape[0])
        self.kernel.grad += np.dot(self.col.T, dy).transpose(1, 0).reshape(*self.kernel.shape)
        dcol = np.dot(dy, self.k_col.T)
        #col2im
        B, C, iH, iW = self.x.shape
        oH, oW = self.z.shape[-2:]
        p, s = self.padding, self.stride
        dcol = dcol.reshape(B, oH, oW, C, self.kernel_size, self.kernel_size).transpose(0, 3, 4, 5, 1, 2)
        dx = np.zeros((B, C, iH + 2*p+s-1, iW + 2*p+s-1))
        for y in np.arange(self.kernel_size):
            for x in np.arange(self.kernel_size):
                dx[:, :, y:y+s*oH:s, x:x+s*oW:s] += dcol[:, :, y, x, :, :]
        if p > 0:
            dx = dx[:, :, p:-p, p:-p]
        return dx.reshape(dx.shape[0],-1)

        # End of todo


class Conv2d_im2col(Conv2d):

    def forward(self, x):
        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.

        B, iC, iH, iW = x.shape
        p, s, k = self.padding, self.stride, self.kernel_size
        oH, oW = (iH-k)//s+1, (iW-k)//s+1
        col = np.zeros((B, iC, k, k, oH, oW))
        for h in np.arange(k):
            for w in np.arange(k):
                col[:, :, h, w, :, :] = x[:, :, h:h+s*oH:s, w:w+s*oW:s]
        return col.transpose(0, 4, 5, 1, 2, 3).reshape(B*oH*oW, -1)
        # B, C, iH, iW = x.shape
        # s, k, p = self.stride, self.kernel_size, self.padding
        # oH, oW = (iH - k) // s + 1, (iW - k) // s + 1
        # st = x.strides
        # st = (*st[:-2], st[-2]*s, st[-1]*s, *st[-2:])
        # # shape指定重新划分后的形状，strides指定对应维度的步幅
        # return np.lib.stride_tricks.as_strided(x, shape=(B, C, oH, oW, k, k), strides=st)

        # k, p, s = self.kernel_size, self.padding, self.stride
        # B, C_in, H_in, W_in = x_padded.shape
        # C_out, H_out, W_out = self.out_channels, (H_in - k) // s + 1, (W_in - k) // s + 1
        # out_size = H_out * W_out
        # col = np.empty((B*H_out*W_out, k**2*C_in))
        # for y in np.arange(H_out):
        #     for x in np.arange(W_out):
        #         col[y*W_out+x::out_size, :] = x[:, :, y*s:y*s+k, x*s:x*s+k].reshape(B, -1)

        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int = 2,
                 stride: int = 2, padding: int = 0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tensor = nn.tensor.zeros(1)  # 保持一致性
        self.x = None
        self.x_padded = None
        self.z = None
        self.tensor = None

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.

        k, s, p = self.kernel_size, self.stride, self.padding
        B, C, H_in, W_in = x.shape
        if p != 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
        else:
            x_padded = x
        self.x_padded = x_padded
        H_out, W_out = int((H_in + 2 * p - k) / s + 1), int((W_in + 2 * p - k) / s + 1)
        self.z = np.zeros((B, C, H_out, W_out))
        for h in np.arange(H_out):
            for w in np.arange(W_out):
                self.z[:, :, h, w] = np.mean(x_padded[:, :, s * h:s * h + k, s * w:s * w + k], axis=(-2, -1))
        return self.z

        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        B, C, H_out, W_out = dy.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        grad = np.zeros_like(self.x_padded)
        for h in np.arange(H_out):
            for w in np.arange(W_out):
                grad[:, :, s * h: s * h + k, s * w: s * w + k] += dy[:, :, h:h + 1, w:w + 1]/k*k
        if p > 0:
            grad = grad[:, :, p:-p, p:-p]
        return grad

        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int = 2,
                 stride: int = 2, padding: int = 0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tensor = nn.tensor.zeros(1)  # 保持一致性
        self.x = None
        self.x_padded = None
        self.z = None
        self.tensor = None
        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.

        k, s, p = self.kernel_size, self.stride, self.padding
        B, C, H_in, W_in = x.shape
        if p != 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
        else:
            x_padded = x
        self.x_padded = x_padded
        H_out, W_out = int((H_in+2*p - k)/s + 1), int((W_in+2*p - k)/s + 1)
        self.z = np.zeros((B, C, H_out, W_out))
        for h in np.arange(H_out):
            for w in np.arange(W_out):
                self.z[:, :, h, w] = np.max(x_padded[:, :, s*h:s*h+k, s*w:s*w+k], axis=(-2,-1))
        return self.z

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.

        B, C, H_out, W_out = dy.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        H_in, W_in = self.x_padded.shape[-2]-2*p, self.x_padded.shape[-1]-2*p
        grad = np.zeros_like(self.x_padded)
        zeros = np.zeros((B, C, s, s))
        for h in np.arange(H_out):
            for w in np.arange(W_out):
                x_part = self.x_padded[:, :, s*h: s*h+k, s*w: s*w+k]
                max_id = np.max(x_part, axis=(2,3))
                grad_part = grad[:, :, s*h: s*h+k, s*w: s*w+k]
                grad_part += np.where(grad_part == np.expand_dims(max_id, (2, 3)),
                                      dy[:, :, h:h+1, w:w+1], zeros)
        if p > 0:
            grad = grad[:, :, p:-p, p:-p]
        return grad
        # End of todo


class Dropout(Module):

    def __init__(self, p: float = 0.8):
        # TODO Initialize the attributes
        # of dropout module.

        self.p = p

        # End of todo

    def forward(self, x):
        # TODO Implement forward propogation
        # of dropout module.

        cast = (np.random.rand(*x.shape)<self.p)/self.p
        return x*cast

        # End of todo

    def backard(self, dy):
        # TODO Implement backward propogation
        # of dropout module.

        ...

        # End of todo


if __name__ == '__main__':
    import pdb;

    pdb.set_trace()
