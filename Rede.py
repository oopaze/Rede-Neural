import numpy as np
import Camada

class Rede(object):
    '''
    Classe responsável pela arquitetura da Rede Neural
    '''
    def __init__(self, numEntradas: int, numNeuronios: int, numCamadas: int, numSaidas: int, pesos = None, bias = None) -> list:
        '''
        As variaveis pesos e bias podem ser fornecidos, se já tiver pesos treinados, ou podem ser inicializados
        sem necessidade do fornecimento, para que seja feito o treinamento.
        '''
        if pesos == None: #Testando se voce passou os pesos a serem usados
            self.camadas = [Camada.Camada(numNeuronios, numEntradas)] #Iniciando camadas com a camada de entrada já inclusa 
        
            for i in range(numCamadas-2): #Criando as camadas ocultas
                self.camadas.append(Camada.Camada(numNeuronios, len(self.camadas[i-1].neuronios)))
            self.camadas.append(Camada.Camada(numSaidas, len(self.camadas[-1].neuronios))) #Criando a camada de saida 
        
        else: 
            self.camadas = [Camada.Camada(numNeuronios, numEntradas, pesos[0], bias[0])] #Iniciando camadas com a camada de entrada já inclusa com os pesos e bias fornecidos
        
            for i in range(numCamadas-2): #Criando as camadas ocultas com os pesos e bias fornecidos
                self.camadas.append(Camada.Camada(numNeuronios, len(self.camadas[i-1].neuronios), pesos[i], bias[i]))
            self.camadas.append(Camada.Camada(numSaidas, len(self.camadas[-1].neuronios), pesos[-1], bias[-1])) #Criando a camada de saida com os pesos e bias fornecidos


        #Pesos, Saidas e Bias usados em todas as camadas para usar no back_propagation
        self.saidas = []
        self.pesos = []
        self.bias = []
        self.aprendizado = 0.1
        
    def feed_foward(self, entradas):
        '''
        Passa os dados em toda rede, gera a quantidade de saidas desejadas e salva os pesos, 
        bias e saidas de cada camada nas variaveis saidas, pesos, bias deste objeto
        '''
        self.saidas, self.pesos, self.bias = [], [], [] #Zerando as saidas, os pesos e os bias gerados na iteração anterior
    
        for i in range(len(self.camadas)):
            saida = self.camadas[i].feed_foward(entradas) #Calculando a saida de cada camada
            self.saidas.append(saida[0]) #Salvando as saidas de camada
            self.pesos.append(saida[1]) #Salvando os pesos de cada camada
            self.bias.append(saida[2]) #Salvando os bias de cada camada
            entradas = saida[0] #Fazendo da saida da camada anterior a entrada da próxima camada
        
        saida = saida[0] #Fazendo da variavel saida somente os dados de saida
        return saida 
    

    def derivada_sig(self, x):
        '''
        Derivada da sigmoid (função de ativação)
        '''
        return x*(1-x)

    def salvar_pesos(self):
        '''
        Função responsavel por salvar pesos já treinando em um documento txt
        '''
        pass

    def back_foward(self, entrada, saida):
        '''
        Funcão responsavel pela revisão dos pesos a cada iteração 
        '''
        pass
 
num_entrada = 2 
num_neuronio = 5
num_camadas = 3
num_saidas = 2

a = Rede(num_entrada, num_neuronio, num_camadas, num_saidas)

a.feed_foward([2,1])


