import math
from platform import java_ver
import random

# klasa mreza
class Net:
    def __init__(self, genome):
        # A unique network can be reconstructed from every genome
        # genome holds this genome in our network
        self.genome = genome
        # layer_num holds the number of layers in our network
        self.layer_num = genome[0]
        # architecture holds an array representing the number of nodes in each layer (input, hidden layers, output)
        self.architecture = []

        self.nodes = []

        # Making the nodes of the network and assining 0 to the value in each node
        for i in range(1, self.layer_num+1):
            self.architecture.append(genome[i])
            self.nodes.append([])

            for j in range(0, self.architecture[i-1]+1):
                self.nodes[i-1].append(0)
 
        self.weights = []
        self.weights.append([])
        idx = self.layer_num+1
        for k in range(1, self.layer_num):
            self.weights.append([])
 
            for i in range(0, self.architecture[k]):
                self.weights[k].append([])
 
                for j in range(0, self.architecture[k-1]+1):
                    self.weights[k][i].append(genome[idx])
                    idx += 1
                    if j == self.architecture[k-1]:
                        self.nodes[k-1][j] = 1

    def Sigmoid(self, value):
        if value > 0:
            return (1 / (1 + math.exp(-value)))
        else:
            return (math.exp(value) / (1 + math.exp(value)))
 
# funkcija koja ce da od vrednosti vrati vrednost izmedju 0 i 1
    def Squish(self, value):
        return self.Sigmoid(value)

# funckija koja unosi prosledjeni niz u prvi sloj neuronske mreze
    def FillInput(self, input):
        for i, x in enumerate(input):
            self.nodes[0][i] = x

# funkcija koja prolazi kroz celu mrezu propagirajuce vrednosti iz prethodnog sloja u trenutni, ostavlja popunjen poslednji sloj na kraju
    def FeedForward(self):
        for k in range(1, self.layer_num):
            for i in range(0, self.architecture[k]):
                self.nodes[k][i] = 0
 
                for j in range(0, self.architecture[k-1]+1):
                    self.nodes[k][i] += self.nodes[k-1][j] * self.weights[k][i][j]
 
                self.nodes[k][i] = self.Squish(self.nodes[k][i])

# funkcija koja vraca vrednosti iz poslednjeg sloja neuronske mreze
    def ReturnOutput(self):
        output = [val for val in self.nodes[self.layer_num-1]]
        output.pop(len(output)-1)
        return output

# funkcija koja uzima niz ulaznih vrednosti za mrezu i vraca niz vrednosti koje mreza da
    def Activate(self, input):
        self.FillInput(input)
        self.FeedForward()
        return self.ReturnOutput()

# funkcija koja generise genom sa nasumicnim tezinama izmedju -30 i 30, kojem se prethodno zada fiksna arhitektura
def GetRandomGenomeFixedArchitecture(layer_num, architecture):
    genome = []
    genome.append(layer_num)
    for arch in architecture:
        genome.append(arch)
    for k in range(1, layer_num):
        for i in range(0, architecture[k]):
            for j in range(0, architecture[k-1]+1):
                genome.append(random.uniform(-30, 30))
    return genome


