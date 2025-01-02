import numpy as np
import pandas as pd


class Utilities:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)


class DataPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def normalize(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / (self.std + 1e-8)

    def denormalize(self, data):
        return data * self.std + self.mean


class CNN:
    def __init__(self, input_channels, output_channels, kernel_size):
        self.kernel = np.random.randn(output_channels, input_channels,
                                      kernel_size, kernel_size) * 0.1
        self.bias = np.zeros(output_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        k_height, k_width = self.kernel.shape[2:]

        out_height = height - k_height + 1
        out_width = width - k_width + 1
        output = np.zeros((batch_size, self.kernel.shape[0],
                           out_height, out_width))

        for b in range(batch_size):
            for c in range(self.kernel.shape[0]):
                for h in range(out_height):
                    for w in range(out_width):
                        output[b, c, h, w] = np.sum(
                            x[b, :, h:h + k_height, w:w + k_width] *
                            self.kernel[c]) + self.bias[c]

        return Utilities.relu(output)


class LSTM:
    def __init__(self, input_size, hidden_size):
        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.h_prev = np.zeros((hidden_size, 1))
        self.c_prev = np.zeros((hidden_size, 1))

    def forward(self, x):
        z = np.row_stack((self.h_prev, x))

        f = Utilities.sigmoid(np.dot(self.Wf, z) + self.bf)
        i = Utilities.sigmoid(np.dot(self.Wi, z) + self.bi)
        c_bar = Utilities.tanh(np.dot(self.Wc, z) + self.bc)

        self.c_prev = f * self.c_prev + i * c_bar

        o = Utilities.sigmoid(np.dot(self.Wo, z) + self.bo)
        self.h_prev = o * Utilities.tanh(self.c_prev)

        return self.h_prev
