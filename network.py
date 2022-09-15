from platform import architecture
import numpy as np

# architecture is an array, the first element is the number of layers in the network (including input and output layer)
# after that there are the number of neruons in each layer in order from input to output layer

class Genome():
    # When constructing genome it will generate parametars, weights and biases, for each layer that are set to be empty
    def __init__(self, architecture, numInputFeatures, batchSize):
        self.architecture = architecture
        self.layers_num = len(architecture)
        self.numInputFeatures = numInputFeatures
        self.batchSize = batchSize
        self.fitness = 0
        self.score = 0
        self.scale = 1
        self.shift = 0
        self.params = {}
        for i in range(1, self.layers_num):
            self.params["W" + str(i)] = np.empty((architecture[i], architecture[i-1])) # mozda je bolje np.zeros
            self.params["b" + str(i)] = np.empty((architecture[i], 1))

    # This function sets the parametars to a random value between 0 and 1 multiplied by "scale". It will also add "shift" to each value
    def SetRandomGenomeUniform(self, scale = 1, shift = 0):
        for i in range(1, self.layers_num):
            self.params["W" + str(i)] = np.random.rand(self.architecture[i], self.architecture[i-1]) * scale + shift
            self.params["b" + str(i)] = np.random.rand(self.architecture[i], 1) * scale + shift
            self.scale = scale
            self.shift = shift

    def SetRandomGenomeNorm(self, scale = 1):
        for i in range(1, self.layers_num):
            self.params["W" + str(i)] = np.random.normal(0, scale, size=(self.architecture[i], self.architecture[i-1]))
            self.params["b" + str(i)] = np.random.normal(0, scale, size=(self.architecture[i], 1))
            self.scale = scale
        

class Net():
    def __init__(self, genome):
        self.genome = genome
        self.architecture = genome.architecture
        self.numInputFeatures = genome.numInputFeatures
        self.batchSize = genome.batchSize
        self.params = genome.params
        self.layers_num = genome.layers_num
        self.cache = {}

    def Sigmoid(self, Z):
        return np.where(Z > 0, 1/(1 + np.exp(-Z)), np.exp(Z) / (1 + np.exp(Z))) # ovo mi daje overflow u exp????

    def ActivationFunction(self, Z):
        return np.tanh(Z)
        #return self.Sigmoid(Z)

    def FeedForward(self, X):
        self.cache["Z1"] = np.dot(self.params["W1"], X)
        self.cache["A1"] = self.ActivationFunction(self.cache["Z1"])
        for i in range(2, self.layers_num):
            self.cache["Z" + str(i)] = np.dot(self.params["W" + str(i)], self.cache["A" + str(i-1)])
            self.cache["A" + str(i)] = self.ActivationFunction(self.cache["Z" + str(i)])

    def Activate(self, X):
        self.FeedForward(X)
        return self.cache["A" + str(self.layers_num-1)]

    
    
#g = Genome([2, 3, 2], 3, 2)
#g.SetRandomGenomeUniform()
#net = Net(g)
#X = [[4, 3], [1, 2], [5, 3]]
#print(net.Activate(X))

