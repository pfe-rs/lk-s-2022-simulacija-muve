import geneticAlgo
import network
import physics as ph
import numpy as np
import matplotlib.pyplot as plt

fitnessGen = []

def main(genomes):
    pe = []
    nets = []
    ge = []
    for _, g in enumerate(genomes):
        p = ph.PhysicsEngine()
        pe.append(p)
        net = network.Net(g)
        nets.append(net)
        g.fitness = 0
        g.score = 0
        ge.append(g)
    
    for i, g in enumerate(ge):
        gR = pe[i].run(nets[i])
        maxFitness = 0
        maxFitnessY = []
        if (gR["checkIfCrashed"]):
            g.fitness = 0
        else:
            g.fitness = np.sum(gR["position"], axis = 0)[2]
            if (g.fitness > maxFitness):
                maxFitness = g.fitness
                maxFitnessY = gR["position"][2]
    plt.plot(maxFitnessY)
    plt.xlable("time")
    plt.ylable("best bug height")
    plt.show()
    fitnessGen.append(maxFitness)
    

        

    
def run():
    p = geneticAlgo.Population(50, [6, 12, 12, 6], scale=30)

    plt.plot(fitnessGen)
    plt.xlable("time")
    plt.ylable("fitness of best bug")
    plt.show()

    p.run(main, 10000, crossover_chance = 4, mutation_chance = 20, score_treshold = 200, delta_score = 200, max_score_treshold = 10000, savefile_prefix = 'vector1-', save_checkpoints = True)

run()