import d2lzh as d2l
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss, nn
from mxnet import gluon, init

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel(name + "(X)")
    d2l.plt.show()

# x = nd.arange(-8.0, 8.0, 0.1)
# x.attach_grad()
# with autograd.record():
#     y = x.relu()
# xyplot(x, y, 'relu')
# y.backward()
# xyplot(x, x.grad, 'grad of rely')

# with autograd.record():
#     y = x.sigmoid()
# xyplot(x, y, 'sigmoid')
#
# y.backward()
# xyplot(x, x.grad, 'grad of sigmoid')

# with autograd.record():
#     y = x.tanh()
# xyplot(x, y, 'tanh')
# y.backward()
# xyplot(x, x.grad, 'grad of tanh')



##Perceptron step by step
## Read the data, define the model, para
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
#
# w1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
# b1 = nd.zeros(num_hiddens)
# w2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
# b2 = nd.zeros(num_outputs)
# params = [w1, b1, w2, b2]
#
# for param in params:
#     param.attach_grad()
#
# def relu(X):
#     return nd.maximum(X, 0)
#
# def net(X):
#     X = X.reshape((-1, num_inputs))
#     H = relu(nd.dot(X, w1) + b1)
#     return nd.dot(H, w2) + b2
#
# loss = gloss.SoftmaxCrossEntropyLoss()
#
# num_epoches, lr = 5, 0.5
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, params, lr)

#Easy way
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epoches = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, None, None, trainer)