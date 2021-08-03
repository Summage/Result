import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import nn
import nn.functional as F

n_features = 28 * 28
n_classes = 10
n_epochs = 100
bs = 1000
lr = 1e-3
lengths = (n_features, 512, n_classes)


class Model(nn.Module):

    # TODO Design the classifier.

    def __init__(self, shape, activation=nn.functional.Sigmoid, affine=nn.functional.Affine,
                 use_dropout=False,
                 use_BN=False, BNFigure=[], use_conv=False, pooling=nn.modules.MaxPool):
        self.original_shape_mnist = (28, 28)
        self.shape = list(shape)
        self.input = None
        self.delta = None
        self.use_BN = use_BN
        self.BNFigure = BNFigure
        self.use_conv = use_conv
        self.pooling = pooling
        if use_dropout is True:
            self.dropout = nn.modules.Dropout()
        else:
            self.dropout = lambda x: x
        self.need_update = [0]
        if self.use_conv is True:
            self.convLayer = [
                nn.BatchNorm1d(self.original_shape_mnist[0] * self.original_shape_mnist[1], *self.BNFigure),
                nn.Conv2d(1, 1), nn.functional.ReLU(), pooling(), nn.functional.Flatten()]
            self.sequential = [affine, activation]
        else:
            self.sequential = [self.initLayer((self.shape[i - 1] + 1, self.shape[i]), affine, activation)
                               for i in range(1, len(shape))]

    def initLayer(self, shape, affine, activation):
        layer = [affine(nn.tensor.from_array(np.random.randn(*shape))), activation()]
        layer[0].tensor[0, :] = 0
        layer[0].tensor[1:, :] = layer[0].tensor[1:, :] / np.sqrt(shape[0])
        # BN
        if self.use_BN:
            self.need_update.append(1)
            layer.insert(0, nn.modules.BatchNorm1d(shape[1], *self.BNFigure))
        return layer

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        if self.use_conv:
            x = np.expand_dims(self.convLayer[0](x).reshape(-1, *self.original_shape_mnist), 1)
            for i in range(1, len(self.convLayer)):
                x = self.convLayer[i].forward(x)
            self.shape[0] = x.shape[-1]
            if not isinstance(self.sequential[0], list):
                self.sequential = [self.initLayer((self.shape[i - 1] + 1, self.shape[i]), *self.sequential)
                                   for i in range(1, len(self.shape))]
        for layer in self.sequential:
            for fn in layer:
                x = fn.forward(x)
            x = self.dropout(x)
        return self.sequential[-1][-1].value

    def setdefault(self):
        # 初始化tensor梯度
        # 更新后重置梯度
        for layer in self.sequential:
            for i in self.need_update:
                layer[i].tensor.grad = np.zeros(layer[i].tensor.shape)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.setdefault()
        delta = self.delta = dy
        if self.use_conv is True:
            sequential = [self.convLayer, *self.sequential]
        else:
            sequential = self.sequential
        for i in range(2, len(sequential[-1]) + 1):
            delta = self.sequential[-1][-i].backward(delta)
        for i in range(2, len(sequential) + 1):
            for j in range(1, len(sequential[-i]) + 1):
                delta = sequential[-i][-j].backward(delta)
        return delta

    # End of todo


def load_mnist(mode='train', n_samples=None):
    images = './train-images.idx3-ubyte' if mode == 'train' else './t10k-images.idx3-ubyte'
    labels = './train-labels.idx1-ubyte' if mode == 'train' else './t10k-labels.idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape((length, 28, 28)).astype(np.int32)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape((length)).astype(np.int32)
    return (X[:n_samples].reshape(n_samples, -1), y[:n_samples]) if n_samples is not None else \
        (X.reshape(length, -1), y)


def vis_demo(model):
    X, y = load_mnist('test', 20)
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    fig = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')[1].flatten()
    for i in range(20):
        img = X[i].reshape(28, 28)
        fig[i].set_title(preds[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("vis.png")
    plt.show()


def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test'))
    model = Model(lengths, use_BN=True, use_conv=True, use_dropout=False)
    optimizer = nn.optim.Adam(model, lr=lr)
    # optimizer = nn.optim.SGD(model, lr=lr, momentum=0.9)
    criterion = F.CrossEntropyLoss(n_classes=n_classes)

    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4 / bs)
        bar.set_description(f'epoch  {i:2}')
        for X, y in bar:
            probs = model.forward(X)
            loss = criterion(probs, y)
            model.backward(loss.backward())
            optimizer.step()
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y) / len(y) * 100:.1f} ')  # loss={loss.value:.3f}

        for X, y in testloader:
            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            print(f' test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

    vis_demo(model)


if __name__ == '__main__':
    main()
