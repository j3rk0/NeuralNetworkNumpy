import numpy as np


def sliding_window_view(img, win_size, stride=1):
    s0, s1 = img.strides  # spazio occupato in memoria da una finestra
    res_size = img.shape[0] - win_size + 1  # dimensione del risultato
    shp = res_size, res_size, win_size, win_size  # shape del risultato
    strd = s0, s1, s0, s1  # strides

    # prendo tutte le finestre e applico lo stride
    return np.lib.stride_tricks.as_strided(img, shape=shp, strides=strd)[::stride, ::stride]


class ConvolutionLayer:

    def __init__(self, in_channels, n_filters, kernel_size, activation):
        self.tag = "convolution"
        self.shape = n_filters

        self.actifun = activation
        self.kernel = .1 * np.random.randn(n_filters, in_channels, kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.in_channels = in_channels
        self.img_size = None

        self._convolved = None
        self._layer_in = None
        self.delta = None
        self.cumuled_delta = np.zeros(self.kernel.shape)

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
        delta *= self.actifun.derivative(self._convolved)  # <- AGGIUNTO
        self.delta = np.zeros(self.kernel.shape)
        ret = np.zeros((self.in_channels, self.img_size, self.img_size))

        for c in range(self.in_channels):
            layer_in_wins = sliding_window_view(self._layer_in[c], self.kernel_size)
            self.delta[c] = np.sum(layer_in_wins.T * delta[c], axis=(2, 3))

            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    ret[:, i:i + self.kernel_size, j:j + self.kernel_size] += delta[c, i, j] * self.kernel[c, :, :, :]
        return ret

    def update(self, eta, mustUpdate=True, batch_size=1):
        self.cumuled_delta += self.delta
        if mustUpdate or batch_size == 1:  # fine del batch oppure online learning
            self.cumuled_delta /= batch_size
            self.kernel -= eta * self.cumuled_delta
            self.cumuled_delta = np.zeros(self.cumuled_delta.shape)

    def reset(self):
        self.kernel = .1 * np.random.rand(self.n_filters, self.in_channels, self.kernel_size, self.kernel_size)


class PaddingLayer:
    def __init__(self, pad):
        self.tag = "padding"
        self.shape = None
        self.pad = pad
        self.img_size = None
        return

    def forw_prop(self, layer_in):
        self.img_size = layer_in.shape[1]
        self.shape = self.img_size + 2 * self.pad
        return np.pad(layer_in, ((0, 0), (self.shape, self.shape), (self.shape, self.shape)))

    def back_prop(self, delta):
        return delta[:, self.shape:self.shape + self.img_size, self.shape:self.shape + self.img_size]

    def update(self, eta, mustUpdate=True, batch_size=1):
        return

    def reset(self):
        return


class PoolingLayer:
    def __init__(self, out_size, mode):
        self.tag = mode + "-pool layer"
        self.shape = out_size
        self.mode = mode
        if mode == "max":
            self.poolfun = np.max
        else:
            self.poolfun = np.mean

        self.stride = None
        self.win_size = None
        self._layer_in = None
        self._pooled = None

    def forw_prop(self, layer_in):
        self._layer_in = layer_in
        channels = layer_in.shape[0]
        img_size = layer_in.shape[1]

        if self.win_size is None:  # calcolo stride e dimensione finestra
            self.stride = img_size // self.shape
            self.win_size = self.stride + img_size % self.shape

        ret = np.zeros((channels, self.shape, self.shape))

        for c in range(channels):
            win = sliding_window_view(layer_in[c, :, :], self.win_size, self.stride)
            ret[c, :, :] = self.poolfun(win, axis=(2, 3))

        self._pooled = ret
        return ret

    def back_prop(self, delta):
        ret = np.zeros(self._layer_in.shape)

        for c in range(self._layer_in.shape[0]):  # for each channel
            for i in range(self.shape):
                for j in range(self.shape):
                    win_i = i * self.stride  # calcolo indici iniziali della finestra
                    win_j = j * self.stride
                    img_win = self._layer_in[c, win_i:win_i + self.shape, win_j:win_j + self.shape]
                    if self.mode == "max":
                        indexes = np.argwhere(img_win == self._pooled[c, i, j])[0]
                        ret[c, indexes[0] + i, indexes[1] + j] = delta[c, i, j]
                    else:
                        ret[c, win_i:win_i + self.shape, win_j:win_j + self.shape] = self._pooled[c, i, j]
        return ret

    def update(self, eta, mustUpdate=True, batch_size=1):
        return

    def reset(self):
        return


class FlattenerLayer:
    def __init__(self, in_channels, in_size):
        self.tag = "flattener"
        self.in_channels = in_channels
        self.in_size = in_size
        self.shape = in_channels * in_size * in_size
        return

    def forw_prop(self, layer_in):
        return layer_in.reshape(self.shape)[..., None]

    def back_prop(self, delta):
        return delta.reshape((self.in_channels, self.in_size, self.in_size))

    def update(self, eta, mustUpdate=True, batch_size=1):
        return

    def reset(self):
        return

# %%
