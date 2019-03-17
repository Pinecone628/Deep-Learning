from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    """
    subclass of Block
    """
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


class MySequential(nn.Block):
    """
    Like Sequential
    """
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x


class FancyMLP(nn.Block):
    """
    More complex model
    """
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        x = self.dense(x)

        while x.norm().asscalar() > 1:
            print(x.norm())
            x /= 2
        if x.norm().asscalar() > 1:
            x *= 10
        return x.sum()


class NestMLP(nn.Block):

    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))


if __name__ == "__main__":
    X = nd.random.uniform(shape=(2, 20))
    # net = MLP()
    # net.initialize()
    # print(net(X))
    #
    # net1 = MySequential()
    # net1.add(nn.Dense(256, activation='relu'))
    # net1.add(nn.Dense(10))
    # net1.initialize()
    # print(net(X))

    # net2 = FancyMLP()
    # net2.initialize()
    # print(net2(X))

    # net3 = nn.Sequential()
    # net3.add(NestMLP(), nn.Dense(20), FancyMLP())
    # net3.initialize()
    # print(net3(X))