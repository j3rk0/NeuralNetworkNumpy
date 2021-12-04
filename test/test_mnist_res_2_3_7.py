import sys

sys.path.extend(['/home/jerko/PycharmProjects/progettoNN'])

from data.MNIST import get_mnist_data
from neural_network.neural_network import NeuralNetwork
from neural_network.layers.convolution_layers import *
from neural_network.layers.dense_layers import *
from neural_network.layers.resilient_layers import *
from neural_network.functions import *

train_X, train_y, valid_X, valid_y, test_X, test_y = get_mnist_data(valid=.25)
test_X = test_X / 255 - .5
train_X = train_X / 255 - .5
valid_X = valid_X / 255 - .5

# %%

net = NeuralNetwork(eta=.1, target_loss=.01, mode='full-batch', loss_fun='cross-entropy')

net.add_all([
    PaddingLayer(2),
    ResilientConvolutionLayer(1, 3, 5, relu),
    PoolingLayer(14, 'max'),
    PaddingLayer(2),
    ResilientConvolutionLayer(3, 7, 5, relu),
    PoolingLayer(7, 'max'),
    FlattenerLayer(7, 7),
    ResilientFullyConnectedLayer(343, 100, relu),
    ResilientFullyConnectedLayer(100, 65, relu),
    ResilientFullyConnectedLayer(65, 10, softMaxCe)
])

# %%
curve = ''
try:
    curve = net.fit(train_X, train_y, valid_X, valid_y)
except KeyboardInterrupt:
    net.save('models/mnist_res_1_3.net')
    np.savetxt('models/mnist_res_1_3.csv', curve, delimiter=',')
# %%
np.savetxt('models/mnist_res_2_3_7.csv', curve, delimiter=',')
net.save('models/mnist_res_2_3_7.net')
