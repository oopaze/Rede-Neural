import numpy as np
from .camada import Camada

class Rede:
    def __init__(self, num_entradas, num_saidas, num_neuronios, num_camadas, learning_hating=0.01, bias=1):
        self.camadas = np.array([])
        self.num_neuronios = num_neuronios
        self.bias = bias
        self.learning_hating = learning_hating

        self.camadas = np.append(self.camadas, Camada(self.bias, num_entradas, self.learning_hating))
        
        if isinstance(num_neuronios, list):
            for i in range(num_camadas-2):
                self.camadas = np.append(self.camadas, Camada(self.bias, self.num_neuronios[i], self.learning_hating))
        
        elif isinstance(num_neuronios, int):
            for i in range(num_camadas-2):
                self.camadas = np.append(self.camadas, Camada(self.bias, self.num_neuronios, self.learning_hating))

        self.camadas = np.append(self.camadas, Camada(self.bias, num_saidas, self.learning_hating))
