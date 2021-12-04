import numpy as np


class InputLayer:
    def __init__(self, input_size):
        self.tag = "input"
        self.shape = input_size
        return

    def forw_prop(self, layer_input):
        return layer_input[..., None]

    def back_prop(self, delta):
        return delta

    def update(self, eta=None, mustUpdate=True, batch_size=1):
        return

    def reset(self):
        return


class FullyConnectedLayer:
    def __init__(self, input_neuroids, n_neuroids, activation):
        self.tag = "hidden"
        self.shape = n_neuroids  # numero neuroni del layer
        self.weights = .1 * np.random.randn(n_neuroids, input_neuroids)  # init pesi
        self.bias = .1 * np.random.randn()  # init bias
        self.actifun = activation  # funzione di attivazione

        # caches [salva i dati per backpropagation e update]
        self._layer_in = None
        self._weighted_in = None
        self._delta = None
        self._cumuled_delta = np.zeros((n_neuroids, input_neuroids))
        self._cumuled_bias_delta = 0

    def forw_prop(self, layer_in):
        self._layer_in = layer_in
        self._weighted_in = self.weights @ layer_in + self.bias  # prodotto matrice-vettore
        return self.actifun(self._weighted_in)

    def back_prop(self, delta):  # ricevo dal layer successivo il delta. Lo moltiplico per la derivata della funz
        # di attivazione
        self._delta = delta * self.actifun.derivative(self._weighted_in)  # gradiente = [ Error'(z) * actifun'(z) ]
        return self.weights.T @ self._delta  # ritorno la trasposta dei pesi per il delta

    def update(self, eta=.1, mustUpdate=True, batch_size=1):

        self._cumuled_delta += self._delta @ self._layer_in.T  # accumula
        self._cumuled_bias_delta += np.sum(self._delta)

        if mustUpdate or batch_size == 1:  # se è online oppure fine del batch
            self._cumuled_delta /= batch_size  # dividi per batch_size
            self._cumuled_bias_delta /= batch_size
            self.weights -= eta * self._cumuled_delta  # aggiorna
            self.bias -= eta * self._cumuled_bias_delta
            self._cumuled_delta = np.zeros(self._cumuled_delta.shape)  # resetta accumulatori
            self._cumuled_bias_delta = 0

    def reset(self):
        self.weights = .1 * np.random.randn(self.weights.shape[0], self.weights.shape[1])