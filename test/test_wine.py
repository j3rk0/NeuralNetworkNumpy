from neural_network.neural_network import NeuralNetwork
from neural_network.layers.dense_layers import *
from neural_network.functions import *
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)

for j in range(X.shape[1]):
    fmax = np.max(X[:, j])
    fmin = np.min(X[:, j])
    X[:, j] = (X[:, j] - fmin) / (fmax - fmin)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
net = NeuralNetwork(eta=.1, target_loss=.01, mode="on-line", loss_fun='cross-entropy')
net.add_all(
    [InputLayer(13),
     FullyConnectedLayer(13, 10, relu),
     FullyConnectedLayer(10, 5, relu),
     FullyConnectedLayer(5, 3, softMaxCe)])
# %%

ret = net.fit(X_train, y_train)


# %%

import matplotlib.pyplot as plt

plt.scatter(x=ret[:, 0], y=ret[:, 1])
plt.show()

# %%

res = net.predict(X_test)
a = accuracy_score(y_test, res)
print(f"accuracy: {a}")
