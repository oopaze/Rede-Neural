import numpy as np

class Neuronio(object):
    def __init__(self, bias, learning_hating):
        self.learning_hating = learning_hating
        self.peso = np.random.randn()
        self.bias = bias

    def feed_foward(self, entrada):
        self.saida = np.multiply(entrada, self.peso) + self.bias
        return self.sig(self.saida)

    def sig(self, x):
        return 1/(1 + np.exp(-x)) 