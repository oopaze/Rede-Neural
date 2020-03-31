import numpy as np        
import Neuronio

class Camada(object):
    '''
    Classe por construir toda a arquitetura das camadas
    '''
    def __init__(self, numNeuronios, numPesos, pesos = None, bias = None):
        self.neuronios = []
        if pesos == None: #Vendo se os pesos foram passados
            self.bias = np.random.randn()
            for e in range(numNeuronios):
                self.neuronios.append(Neuronio.Neuronio(numPesos, self.bias))
        else: #Adicionando pesos passados aos neuronios
            self.bias = bias 
            for e in range(numNeuronios):
                self.neuronios.append(Neuronio.Neuronio(numPesos, self.bias, pesos[e]))

        self.saidas = []
        self.pesos = []


    def feed_foward(self, entradas):
        """
        Calcula a saida de cada neuronio e salva os pesos usados
        """
        self.saidas = []
        self.pesos = []
        for i in range(len(self.neuronios)):
            self.saidas.append(self.neuronios[i].feed_foward(entradas)[0])
            self.pesos.append(self.neuronios[i].feed_foward(entradas)[1])
        
        return (self.saidas, self.pesos, self.bias)
        

