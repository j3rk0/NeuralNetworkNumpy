import numpy as np


def sliding_window_view(img, win_size, stride=1):
    s0, s1 = img.strides  # spazio occupato in memoria da una finestra
    res_size = img.shape[0] - win_size + 1  # dimensione del risultato
    shp = res_size, res_size, win_size, win_size  # shape del risultato
    strd = s0, s1, s0, s1  # strides

    # prendo tutte le finestre e applico lo stride
    return np.lib.stride_tricks.as_strided(img, shape=shp, strides=strd)[::stride, ::stride]


class ResilientFullyConnectedLayer:
    def __init__(self, input_neuroids, n_neuroids, activation, dmax=50):
        self.tag = "hidden"
        self.shape = n_neuroids  # numero neuroni del layer
        self.weights = .1 * np.random.randn(n_neuroids, input_neuroids)  # init pesi
        self.bias = .1 * np.random.randn()  # init bias
        self.actifun = activation  # funzione di attivazione

        self._delta = None
        self._last_delta = None
        self._last_bias_delta = None
        self._cumuled_delta = np.zeros(self.weights.shape)
        self._cumuled_bias_delta = 0
        self._d = None
        self._bd = None

        self.dmin = 1e-6
        self.dmax = dmax
        self.npos = 1.2
        self.nneg = .5

        # caches [salva i dati per backpropagation e update]
        self._layer_in = None
        self._weighted_in = None

    def forw_prop(self, layer_in):
        self._layer_in = layer_in
        self._weighted_in = self.weights @ layer_in + self.bias  # prodotto matrice-vettore
        return self.actifun(self._weighted_in)

    def back_prop(self, delta):  # ricevo dal layer successivo il delta. Lo moltiplico per la derivata della funz
        # di attivazione
        self._delta = delta * self.actifun.derivative(self._weighted_in)  # gradiente = [ Error'(z) * actifun'(z) ]
        return self.weights.T @ self._delta  # ritorno la trasposta dei pesi per il delta

    def update(self, d0=.1, mustUpdate=True, batch_size=1):

        self._cumuled_delta += self._delta @ self._layer_in.T  # accumulo i delta
        self._cumuled_bias_delta += np.sum(self._delta)

        if mustUpdate:  # fine batch
            if self._last_delta is None:  # prima iterazione
                self._last_bias_delta = 0
                self._last_delta = np.zeros(self._cumuled_delta.shape)
                self._bd = d0
                self._d = np.ones(self._cumuled_delta.shape) * d0

            # AGGIORNO BIAS
            if self._last_bias_delta * self._cumuled_bias_delta > 0:  # stesso segno
                self._bd = np.minimum(self._bd * self.npos, self.dmax)
                self.bias -= np.sign(self._cumuled_bias_delta) * self._bd
                self._last_bias_delta = self._cumuled_bias_delta
            elif self._last_bias_delta * self._cumuled_bias_delta < 0:  # diverso segno
                self._bd = np.maximum(self._bd * self.nneg, self.dmin)
                self._last_bias_delta = 0
            else:  # zero
                self.bias -= np.sign(self._cumuled_bias_delta) * self._bd
                self._last_bias_delta = self._cumuled_bias_delta

            # AGGIORNO PESI
            # stesso segno
            same_sign = self._cumuled_delta * self._last_delta > 0
            self._d[same_sign] = np.minimum(self._d[same_sign] * self.npos, self.dmax)
            self.weights[same_sign] -= np.sign(self._cumuled_delta[same_sign]) * self._d[same_sign]
            self._last_delta[same_sign] = self._cumuled_delta[same_sign]

            # diverso segno
            diff_sign = self._cumuled_delta * self._last_delta < 0
            self._d[diff_sign] = np.maximum(self._d[diff_sign] * self.nneg, self.dmin)
            self._last_delta[diff_sign] = 0

            # zero
            no_sign = self._cumuled_delta * self._last_delta == 0
            self.weights[no_sign] -= np.sign(self._cumuled_delta[no_sign]) * self._d[no_sign]
            self._last_delta[no_sign] = self._cumuled_delta[no_sign]

            self._cumuled_delta = np.zeros(self.weights.shape)  # resetto accumulatori
            self._cumuled_bias_delta = 0

    def reset(self):
        self.weights = .1 * np.random.randn(self.weights.shape[0], self.weights.shape[1])


class ResilientConvolutionLayer:

    def __init__(self, in_channels, n_filters, kernel_size, activation, dmax=50):
        self.tag = "convolution"
        self.shape = n_filters

        self.actifun = activation  # <- AGGIUNTO
        self.kernel = .1 * np.random.randn(n_filters, in_channels, kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.n_filters = n_filters  # <- AGGIUNTO
        self.in_channels = in_channels  # <- AGGIUNTO
        self.img_size = None

        self._convolved = None
        self._layer_in = None
        self._delta = None
        self._cumuled_delta = np.zeros(self.kernel.shape)
        self._last_delta = None
        self._d = None
        self.dmin = 1e-6
        self.dmax = dmax
        self.npos = 1.2
        self.nneg = .5

    def forw_prop(self, layer_in):
        self._layer_in = layer_in
        self.img_size = layer_in.shape[1]
        self.shape = layer_in.shape[1] - self.kernel_size + 1
        ret = np.zeros((self.n_filters, self.shape, self.shape))

        for f in range(self.n_filters):
            for c in range(self.in_channels):
                # prendo le finestre
                sw = sliding_window_view(layer_in[c, :, :], self.kernel_size)
                res = sw * self.kernel[f, c, :, :]  # le moltiplico  per il kernel
                ret[f, :, :] += np.sum(res, axis=(2, 3))  # prendo la somma di ogni finestra

        self._convolved = ret
        return self.actifun(self._convolved)

    def back_prop(self, delta):
        delta *= self.actifun.derivative(self._convolved)
        self._delta = np.zeros(self.kernel.shape)
        ret = np.zeros((self.in_channels, self.img_size, self.img_size))

        for c in range(self.in_channels):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    ret[:, i:i + self.kernel_size, j:j + self.kernel_size] += delta[c, i, j] * self.kernel[c, :, :, :]
        return ret

    def update(self, d0, mustUpdate=True, batch_size=1):
        self._cumuled_delta += self._delta

        if mustUpdate:  # fine del batch oppure online learning
            if self._last_delta is None:  # prima iterazione
                self._last_delta = np.zeros(self._cumuled_delta.shape)
                self._d = np.ones(self._cumuled_delta.shape) * d0

            # AGGIORNO PESI
            # stesso segno
            same_sign = self._cumuled_delta * self._last_delta > 0
            self._d[same_sign] = np.minimum(self._d[same_sign] * self.npos, self.dmax)
            self.kernel[same_sign] -= np.sign(self._cumuled_delta[same_sign]) * self._d[same_sign]
            self._last_delta[same_sign] = self._cumuled_delta[same_sign]

            # diverso segno
            diff_sign = self._cumuled_delta * self._last_delta < 0
            self._d[diff_sign] = np.maximum(self._d[diff_sign] * self.nneg, self.dmin)
            self._last_delta[diff_sign] = 0

            # zero
            no_sign = self._cumuled_delta * self._last_delta == 0
            self.kernel[no_sign] -= np.sign(self._cumuled_delta[no_sign]) * self._d[no_sign]
            self._last_delta[no_sign] = self._cumuled_delta[no_sign]

            self._cumuled_delta = np.zeros(self.kernel.shape)  # resetto accumulatori

    def reset(self):
        self.kernel = .1 * np.random.rand(self.n_filters, self.in_channels, self.kernel_size, self.kernel_size)
