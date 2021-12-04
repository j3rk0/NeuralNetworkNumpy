from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from neural_network.neural_network import NeuralNetwork
from neural_network.layers.convolution_layers import *
from neural_network.layers.dense_layers import *
from neural_network.functions import *

# %%
X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# trasformo le feature in immagini
data_train = []
data_test = []
for i in range(x_train.shape[0]):
    data_train.append(x_train[i].reshape(1, 8, 8))
for i in range(x_test.shape[0]):
    data_test.append(x_test[i].reshape(1, 8, 8))

data_train = np.array(data_train)
data_train /= 16 - .5
data_test = np.array(data_test)
data_test /= 16 - .5
# %%

net = NeuralNetwork(eta=.05, target_loss=.01, mode='mini-batch', batch_size=5, loss_fun='cross-entropy')
net.add_all([PaddingLayer(2),
             ConvolutionLayer(in_channels=1, n_filters=3, kernel_size=3, activation=tanh),
             FlattenerLayer(3, 10),
             FullyConnectedLayer(300, 150, relu),
             FullyConnectedLayer(150, 50, relu),
             FullyConnectedLayer(50, 10, softMaxCe)
             ])

# %%

ret = net.fit(data_train, y_train)


# %%


import matplotlib.pyplot as plt

plt.scatter(x=ret[:, 0], y=ret[:, 1])
plt.show()

# %%

pred = net.predict(data_test)
acc = accuracy_score(y_test, pred)
print(f"accuracy: {acc}")
