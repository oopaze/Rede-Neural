import numpy as np


class Neuronio(object):
    """
    Classe responsavel pela arquitetura dos neuronios
    """
    def __init__(self, entradas: list, bias: int, pesos = None):
        self.bias = bias #Adicionando Bias da camada ao neuronio
        
        if pesos == None: #Testando se nenhum pesos pre treinado foi passado
            self.pesos = np.random.randn(entradas) #Gerando pesos aleatorios
        else:
            self.pesos = pesos #Atribuindo pesos pre treinados a peso

        self.saida = 0 #Somador da saida

    def feed_foward(self, entradas):
        '''
        Função responsavel pelo fluxo de cálculo de cada peso, pela respectiva entrada
        '''
        saida = 0 #Inicializando a saida
        for i in range(len(entradas)): #somando a saida cada entrada vezes seu peso recpectivo
            saida += entradas[i] * self.pesos[i]
        saida += self.bias #Adicionando o bias a saida
        self.saida = self.sig(saida) #Ativando a saida por meio da função de ativação

        return (self.saida, self.pesos) #returnando a saida obtida e os pesos usados para geração dessa saida
    
    def sig(self, x):
        '''
        Função resposável pela a ativação da saida
        '''
        return 1/(1 + np.exp(-x)) #Função de ativação do tipo SIGMOID
