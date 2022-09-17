import geneticAlgo
import network
import physics as ph
import numpy as np
import matplotlib.pyplot as plt
import pickle

fitnessGen = []
genNum = 0

class Bug:
    def __init__(self, gR, fitness):
        self.bugParams = gR
        self.fitness = fitness
    def CompareFunction(self, b):
        return b.fitness

def main(genomes):
    global genNum, fitnessGen
    genNum += 1
    pe = []
    nets = []
    ge = []
    bugs = []
    for _, g in enumerate(genomes):
        p = ph.PhysicsEngine()
        pe.append(p)
        net = network.Net(g)
        nets.append(net)
        g.fitness = 0
        g.score = 0
        ge.append(g)
    
    maxFitness = -100000
    height = []
    for i, g in enumerate(ge):
        #print("  " + str(i))
        gR = pe[i].run(nets[i],symetricWings = True)
        if (gR["checkIfCrashed"]):
            print("Crashed!")
            g.fitness = 0
        else:
            #g.fitness = np.sum(1 - np.abs(gR["position"][2]), axis = 0)
            g.fitness = np.sum(gR["position"][2]/3000, axis = 0)
            if (g.fitness > maxFitness):
                maxFitness = g.fitness
                height = gR["position"][2]
        bugs.append(Bug(gR, g.fitness))
    if(genNum % 10 == 0):
        plt.plot(height)
        plt.xlabel("time")
        plt.ylabel("best bug height")
        plt.show()
    
    bugs.sort(key = bugs[0].CompareFunction, reverse = True)

    if(genNum % 10 == 0):
        savefile_name = 'generation-' + str(genNum) + '-bestBug4'
        with open(savefile_name, "wb") as f:
            pickle.dump(bugs, f)
    fitnessGen.append(maxFitness)
    
def run():
    p = geneticAlgo.Population(50, [15, 20, 20, 3], scale=30)

    p.run(main, 100, crossover_chance = 4, mutation_chance = 5, savefile_suffix='30g-3', score_treshold = 200, delta_score = 200, max_score_treshold = 10000, savefile_prefix = 'bug1-', save_checkpoints = True)

    plt.plot(fitnessGen)
    plt.xlabel("generations")
    plt.ylabel("fitness of best bug")
    plt.show()
if __name__ == "__main__":
    run()