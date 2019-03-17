import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import loss as gloss, nn
import time


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n


def train_ch5(net, train_iter, test_iter, batchi_size, trainer, ctx, num_epoches):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epoches):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, trian acc %.3f, test acc %.3f,''time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))


if __name__ == "__main__":
    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10))

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    ctx = try_gpu()
    print(ctx)

    lr, num_epoches = 0.5, 20
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epoches)


