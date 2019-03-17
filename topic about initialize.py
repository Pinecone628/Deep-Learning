from mxnet import init, nd
from mxnet.gluon import nn


class MyInit(init.Initializer):
    """Def own initializer"""
    def _init_weight(self, name, data):
        print('Init', name, data)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


if __name__ == "__main__":
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    net.initialize()

    X = nd.random.uniform(shape=(2, 20))
    Y = net(X)

    # print(net[1].params)
    # print(net[1].weight.data())
    # print(net[1].weight.grad())
    # print(net.collect_params())
    # print(net.collect_params('.*weight'))

    # net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
    # print(net[0].weight.data()[0])
    #
    # net.initialize(init=init.Constant(1), force_reinit=True)
    # print(net[0].weight.data()[0])
    #
    # net[0].initialize(init=init.Xavier(), force_reinit=True)
    # print(net[0].weight.data()[0])
    #
    # net[0].initialize(MyInit(), force_reinit=True)
    # print(net[0].weight.data()[0])

    #Share the para
    net2 = nn.Sequential()
    shared = nn.Dense(8, activation='relu')
    net2.add(nn.Dense(8, activation='relu'),
             shared,
             nn.Dense(8, activation='relu', params=shared.params),
             nn.Dense(10))
    net2.initialize()
    print(net2(X))

    X = nd.random.uniform(shape=(2, 20))
    print(net2(X))