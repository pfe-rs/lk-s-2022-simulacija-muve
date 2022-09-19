import geneticAlgo
import network
import physics as ph
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

fitnessGen = []
genNum = 0
startTime = 0
prevTime = 0

class Bug:
    def __init__(self, gR, fitness):
        self.bugParams = gR
        self.fitness = fitness
    def CompareFunction(self, b):
        return b.fitness

def main(genomes):
    global genNum, fitnessGen, startTime, prevTime
    genNum += 1
    pe = []
    nets = []
    ge = []
    bugs = []
    for _, g in enumerate(genomes):
        p = ph.PhysicsEngine()
        pe.append(p)
        net = network.Net(g.genome)
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
            timePoints = 3000
            #g.fitness = np.sum(gR["position"][2]/timePoints, axis = 0)
            #g.fitness -= np.sum(gR["position"][0]/timePoints, axis = 0)/5
            #g.fitness -= np.sum(gR["position"][1]/timePoints, axis = 0)/5
            g.fitness  = gR["position"][2][-1] # Reminder - [-1] je zadnji element niza
            if (g.fitness > maxFitness):
                maxFitness = g.fitness
                height = gR["position"][2]
        bugs.append(Bug(gR, g.fitness))

    #########################################################################
    ######################    PRINTING AND SAVING    ########################
    #########################################################################

    # printing every x-th generations graph
    
    x = 25
    if(genNum % x == 0):
        plt.plot(height)
        plt.xlabel("time")
        plt.ylabel("best bug height")
        plt.show()
    
    bugs.sort(key = bugs[0].CompareFunction, reverse = True)

    # saving every x-th generation
    x = 10
    if(genNum % x == 0):
        savefile_name = 'generation-' + str(genNum) + '-bestBug5'
        with open(savefile_name, "wb") as f:
            pickle.dump(bugs, f)
    fitnessGen.append(maxFitness)

    # printing average time per generation over last x generations and managing time
    
    currTime = time.time()
    x = 1
    print("Gen " + str(genNum) + " avg. time: " + str((currTime - prevTime)/x))
    prevTime = currTime
    
    

def run():
    global startTime, prevTime
    p = geneticAlgo.Population(100, 4, [15, 20, 20, 3])

    startTime = time.time()
    prevTime = startTime

    totalGenerations = 50
    p.run(main, totalGenerations, score_treshold = 200, delta_score = 200, max_score_treshold = 10000, savefile_prefix = 'bug1-', save_checkpoints = True)
    currTime = time.time()
    print("TOTAL TIME: " + str(currTime - startTime))
    print("AVERAGE TIME PER GENERATION: " + str((currTime - startTime)/totalGenerations))
    plt.plot(fitnessGen)
    plt.xlabel("generations")
    plt.ylabel("fitness of best bug")
    plt.show()
    
if __name__ == "__main__":
    run()