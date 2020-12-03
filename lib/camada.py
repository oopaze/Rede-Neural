import numpy as np
from .neuronio import Neuronio

class Camada:
    def __init__(self, bias, num_neuronios, learning_hating):
        self.learning_hating = learning_hating
        self.bias = bias
        self.neuronios = np.array([])

        for i in range(num_neuronios):
            self.neuronios = np.append(self.neuronios, Neuronio(self.bias, self.learning_hating))
        