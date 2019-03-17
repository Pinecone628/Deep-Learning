from mxnet import nd, autograd
from IPython import display
from matplotlib import pyplot as plt
import random
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet import init
from mxnet import gluon

# Step by step way
# Create data
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# add residual
labels += nd.random.normal(scale=0.01, shape=labels.shape)


# show the relationship between features and label
def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()


# read data and use mini-batch to randomly select
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # break the order of indices
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)

# Show the first batch
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# initialize the parameters and take derivation
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1, ))

w.attach_grad()
b.attach_grad()

# Set the model, loss function and optimize algorithm
def linreg(X, w, b):
    return nd.dot(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# Train the example
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

print(true_w, w, true_b, b)







# Use the gluon
# Use the same parameters

# batch the data
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
for X,y in data_iter:
    print(X, y)
    break

# def the model, loss, optimize and initialize
net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init.Normal(sigma=0.01))

loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

#Train the model

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch: %d, loss: %f' % (epoch, l.mean().asnumpy()))

dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())
