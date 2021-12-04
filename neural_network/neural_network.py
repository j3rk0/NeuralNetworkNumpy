import gzip
from neural_network.functions import *
import pickle
import pickletools
import numpy as np
from time import time
import sys


class NeuralNetwork:

    def __init__(self, eta=.1, target_loss=None, target_epochs=None, target_acc=None, mode='on-line', batch_size=1,
                 loss_fun='sum-of-square'):
        self.layers = []
        self.eta = eta  # learning rate  0 < eta <= 1
        assert (target_loss is not None or target_epochs is not None or target_acc is not None)
        # almeno una condizione di stop ci deve essere
        self.target_epochs = target_epochs
        self.target_loss = target_loss
        self.target_acc = target_acc
        self.mode = mode  # [full-batch/mini-batch/on-line]
        self.loss_metric = loss_fun
        self.learn_curve = []
        self.curr_loss = None

        if mode == 'on-line':
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        if loss_fun == 'sum-of-square':
            self.loss_fun = sumOfSquare
        else:
            self.loss_fun = crossEntropySM

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_all(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def fit(self, X, y, valid_X=None, valid_y=None):
        n_sample = X.shape[0]

        if self.mode == 'full-batch':
            self.batch_size = n_sample

        target = self._one_hot(y)

        epoch = 0
        train_acc = 0
        valid_acc = 0
        err = 1
        start = time()
        ret = []
        try:
            while (self.target_epochs is None or self.target_epochs > epoch) and \
                    (self.target_loss is None or err > self.target_loss) and \
                    (self.target_acc is None or train_acc < self.target_acc):  # fino alla condizione di stop
                epoch += 1
                err = 0
                for i in range(n_sample):  # per ogni dato nel t.s.
                    # bisogna aggiornare i pesi se ci troviamo alla fine del batch
                    to_update = (i % self.batch_size + 1 == self.batch_size)
                    self._back_prop(X[i, :], target[i, :], self.batch_size, to_update)

                    err += np.sum(np.abs(self.curr_loss))
                    sys.stdout.write(f"epoch {epoch} processing sample {i + 1}/{n_sample} curr loss:{err / (i + 1)}\r")
                    sys.stdout.flush()

                err /= n_sample
                print()
                sys.stdout.write(f"calculating training accuracy...\r")
                sys.stdout.flush()

                train_acc = (n_sample - np.count_nonzero(self.predict(X) - y)) / n_sample  # accuracy
                print(f"epoch {epoch} training accuracy: {train_acc}")
                if valid_X is not None:
                    sys.stdout.write(f"calculating validation accuracy...\r")
                    sys.stdout.flush()
                    valid_acc = (valid_X.shape[0] - np.count_nonzero(self.predict(valid_X) - valid_y)) / valid_X.shape[
                        0]
                    print(f"epoch {epoch} validation accuracy: {valid_acc}")

                ret.append({'epoch': epoch, 'loss': err, 'train_accuracy': train_acc,
                            'validation_accuracy': valid_acc})  # dati sul training
        except KeyboardInterrupt:
            print("\ntraining stopped by user\n")

        print(f"elapsed time: {time() - start} s")
        return np.array(ret)

    def predict(self, x):
        ret = []
        for i in range(x.shape[0]):
            ret.append(np.argmax(self._forw_prop(x[i, :])))
        return np.array(ret)

    def _forw_prop(self, x):  # propago x per ogni layer
        for layer in self.layers:
            x = layer.forw_prop(x)
        return x.T

    def _back_prop(self, x, t, batch_size, to_update=False):  # se necessario, upgrado i pesi. Per aggiornarli mi serve
        curr = x  # batch_size.
        for layer in self.layers:  # propaga avanti
            curr = layer.forw_prop(curr)

        self.curr_loss = self.loss_fun(curr, t[..., None])  # calcolo perdita
        curr = self.loss_fun.derivative(curr, t[..., None])  # calcolo gradiente

        if self.loss_metric == 'cross-entropy':  # normalizza perdita
            self.curr_loss /= np.log(t.shape[0])

        for layer in reversed(self.layers):  # calcola i delta
            curr = layer.back_prop(curr)

        for layer in self.layers:  # aggiorna
            layer.update(self.eta, to_update, batch_size)

    def _one_hot(self, y):  # 1 in corrispondenza della 'parola', 0 altrimenti
        n_class = np.max(y) + 1
        oh = np.zeros((y.shape[0], n_class))
        for i in range(y.shape[0]):
            oh[i, y[i]] = 1
        return oh

    def print(self):
        for i in range(len(self.layers)):
            print(f" {i}) {self.layers[i].tag} layer shape {self.layers[i].shape}")

    # salvo su un file la rete addestrata così da non perdere tempo a riaddestrarla. La ricarico da load()
    def save(self, filepath):
        with gzip.open(filepath, "wb") as f:
            pickled = pickle.dumps(self)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)

    @staticmethod
    def from_file(filepath):
        with gzip.open(filepath, 'rb') as f:
            p = pickle.Unpickler(f)
            return p.load()

    def reset(self):
        for layer in self.layers:
            layer.reset()
