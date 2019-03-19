import d2lzh as d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time
import zipfile


class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)




with zipfile.ZipFile('/Users/wujiawei/Documents/d2l-zh/data/jaychou_lyrics.txt.zip') as zin:
    with zin.open("jaychou_lyrics.txt") as f:
        corpus_chars = f.read().decode('utf-8')

corpus_chars = corpus_chars.replace('\n', '').replace('\r', '')
corpus_chars = corpus_chars[0:10000]

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
corpus_indices = [char_to_idx[char] for char in corpus_chars]

num_hiddens, batch_size = 256, 2
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
state = rnn_layer.begin_state(batch_size=batch_size)

num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)


