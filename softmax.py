import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
from mxnet import autograd, nd
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

# Get the data
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
print(len(mnist_test), len(mnist_train))
feature, label = mnist_train[0]
print(feature.shape, feature.dtype)
print(label, type(label), label.dtype)

#show the plots
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    d2l.plt.show()

# show the plots
# X, y = mnist_train[0:9]
# # show_fashion_mnist(X, get_fashion_mnist_labels(y))

# read the batch data
batch_size = 256
# transformer = gdata.vision.transforms.ToTensor()
# if sys.platform.startswith('win'):
#     num_workers = 0
# else:
#     num_workers = 4
#
# train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
#                               batch_size, shuffle=True, num_workers=num_workers)
# test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
#                              batch_size, shuffle=False, num_workers=num_workers)
#
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# initialize the para
num_inputs = 784 #28*28
num_output = 10

w = nd.random.normal(scale=0.01, shape=(num_inputs, num_output))
b = nd.zeros(num_output)

w.attach_grad()
b.attach_grad()

# set model, loss, accuracy,
def softmac(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition


def net(X):
    return softmac(nd.dot(X.reshape((-1, num_inputs)), w) + b)


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return  acc_sum / n

# train the model
num_epoches, lr = 5, 0.1
def train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epoches):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y= y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, trian acc%.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epoches, batch_size, [w,b], lr)

#predict
for X, y in test_iter:
    break
true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
#d2l.show_fashion_mnist(X[0:20], titles[0:20])
#d2l.plt.show()

# Gluon
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
num_epoches = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, None, None, trainer)