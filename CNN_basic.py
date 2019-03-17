from mxnet import autograd, nd
from mxnet.gluon import nn
import d2lzh as d2l


def corr2d(X, K):
    """
    2-way cross-correlation
    :param X: input
    :param K: kernel
    :return: output
    """
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Block):
    """
    2 way convolution layer
    """
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1, ))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


def comp_conv2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


def corr2d_muli_in(X, K):
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])

def corr2d_muli_in_out(X, K):
    return nd.stack(*[corr2d_muli_in(X, k) for k in K])

def corr2d_muli_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_0 = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_0, c_i))
    Y = nd.dot(K, X)
    return Y.reshape((c_0, h, w))

def pool2d(X, pool_size, mode = 'max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j +p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

if __name__ == "__main__":
    # X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # K = nd.array([[0, 1], [2, 3]])
    # print(corr2d(X, K))
    #
    # X = nd.ones((6, 8))
    # X[:, 2:6] = 0
    # print(X)
    # K = nd.array([[1, -1]])
    # Y = corr2d(X, K)
    # print(Y)
    #
    # X2 = nd.ones((6, 8))
    # X2[2:4, :] = 0
    # print(X2)
    # K2 = nd.array([[1], [-1]])
    # print(K, corr2d(X2, K2))

    # conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
    # X = nd.random.uniform(shape=(8, 8))
    # print(comp_conv2d(conv2d, X).shape)
    #
    # conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
    # print(comp_conv2d(conv2d, X).shape)

    # X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    #               [[1, 2, 3], [4, 5, 6], [7, 8 ,9]]])
    # K = nd.array([[[0, 1], [2, 3]],
    #               [[1, 2], [3, 4]]])
    # print(X, K, corr2d_muli_in(X, K))
    # K = nd.stack(K, K+1, K+2)
    # print(corr2d_muli_in_out(X, K))

    # X = nd.random.uniform(shape=(3, 3, 3))
    # K = nd.random.uniform(shape=(2, 3, 1, 1))
    # print(corr2d_muli_in_out_1x1(X, K))
    # print(corr2d_muli_in_out(X, K))
    # X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # print(pool2d(X, (2, 2)))
    # print(pool2d(X, (2, 2), mode='avg'))

    # X = nd.arange(16).reshape((2, 2, 2, 2))
    # print(X)
