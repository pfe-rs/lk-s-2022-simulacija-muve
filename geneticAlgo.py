import network
import random
import pickle
import numpy as np

class Population:
    def __init__(self, population_size, architecture, scale = 1, shift = 0):
        self.scale = scale
        self.shift = shift
        self.crossover_chance = 2
        self.mutation_chance = 10
        self.population_size = population_size
        self.architecture = architecture
        self.layer_num = len(architecture)
        # Creating a population of the population_size with random genomes with fixed architecutres
        self.population = []
        for i in range(0, self.population_size):
            g = network.Genome(architecture, architecture[0], 1)
            #network.Genome.SetRandomGenomeUniform(g, scale = scale, shift = shift)
            network.Genome.SetRandomGenomeNorm(g, scale = scale)
            self.population.append(g)
        self.generationIndex = 1

    # Function that returns the fitness value for the genome, used for the sort function
    def compareFunction(self, g):
        return g.fitness
    
    # Function that takes the best 10 percent of a population and puts it into a mating pool
    # It selects random genomes from the pool and goes through the genome and creates a new genome
    # Every value of the new genome has a 50% chance to be from the first gene and 50% chance to be from the second gene
    # The architecture of the network won't change
    # There is a 50% chance for each genome to be mutated
    # If a genome is chosen to be mutated, max(1, 10%) of its values will be mutated, architecture won't change
    # The % of changed values may be smaller because the same value can be chosen to mutate multiple times
    
    # This function works the same as the one before, but it saves the top max(1, 5%) of the old population into the new one
    def MatingPool10percentRandomIndexMergeSaveTop5percent(self):
        # Creating a mating pool with the best 10% of the genomes
        matingPool = []
        for i in range(0, self.population_size//10):
            matingPool.append(self.population[i])
        self.population.clear()

        # Calculating the number of genomes that will be saved from the currnet population
        save_num = max(self.population_size//20, 1)

        # Finding random genomes from the mating pool to crossover and mutate the resulting genome
        for _ in range(0, self.population_size-save_num):
            # Picking the random genomes
            index1 = random.randrange(0, len(matingPool))
            index2 = random.randrange(0, len(matingPool))

            # Creating and filling the new genome
            newGenome = network.Genome(self.architecture, self.architecture[0], self.population_size)
            # Adding random values from one of the two selected genomes from the mating pool
            for k in range(1, self.layer_num):
                newGenome.params["W" + str(k)] = np.where(np.random.randint(self.crossover_chance, size=(self.architecture[k], self.architecture[k-1])) == 0,
                                                          matingPool[index1].params["W" + str(k)],
                                                          matingPool[index2].params["W" + str(k)])
                newGenome.params["b" + str(k)] = np.where(np.random.randint(self.crossover_chance, size=(self.architecture[k])) == 0,
                                                          matingPool[index1].params["b" + str(k)],
                                                          matingPool[index2].params["b" + str(k)])
            # Mutating the new genome
            for k in range(1, self.layer_num):
                newGenome.params["W" + str(k)] = np.where(np.random.randint(self.mutation_chance, size=(self.architecture[k], self.architecture[k-1])) == 0,
                                                          newGenome.params["W" + str(k)],
                                                          #np.random.rand(self.architecture[k], self.architecture[k-1]) * newGenome.scale + newGenome.shift)
                                                          np.random.normal(0, newGenome.scale, size=(self.architecture[k], self.architecture[k-1])))
                newGenome.params["b" + str(k)] = np.where(np.random.randint(self.mutation_chance, size=(self.architecture[k])) == 0,
                                                          newGenome.params["b" + str(k)],
                                                          #np.random.rand(self.architecture[k]) * newGenome.scale + newGenome.shift)
                                                          np.random.normal(0, newGenome.scale, size=(self.architecture[k])))

            # Adding the new genome to the population
            self.population.append(newGenome)

        # Adding the top 5% of the previous population
        for idx in range(self.population_size-save_num, self.population_size):
            self.population.append(matingPool[idx - (self.population_size-save_num)])


    def FlattenParams(self, g):
        flat = np.empty(0)
        for k in range(1, g.layers_num):
            flat = np.concatenate((flat, np.reshape(g.params["W" + str(k)], (-1))))
            flat = np.concatenate((flat, np.reshape(g.params["b" + str(k)], (-1))))
        return flat
    
    def UnflattenParams(self, flat, g):
        idx = 0
        for k in range(1, g.layers_num):
            for i in range(g.architecture[k]):
                for j in range(g.architecture[k-1]):
                    g.params["W" + str(k)][i][j] = flat[idx]
                    idx += 1
            for i in range(g.architecture[k]):
                g.params["b" + str(k)][i] = flat[idx]
                idx += 1


    def MatingPool10percentRandomTwoPointCrossoverMergeSaveTop5percent(self):
        # Creating a mating pool with the best 10% of the genomes
        matingPool = []
        for i in range(0, self.population_size//10):
            matingPool.append(self.population[i])
        self.population.clear()

        # Calculating the number of genomes that will be saved from the currnet population
        save_num = max(self.population_size//20, 1)

        # Finding random genomes from the mating pool to crossover and mutate the resulting genome
        for _ in range(0, self.population_size-save_num):
            # Picking the random genomes
            index1 = random.randrange(0, len(matingPool))
            index2 = random.randrange(0, len(matingPool))

            # Creating offspring / new genome
            newGenome = network.Genome(self.architecture, self.architecture[0], self.population_size)
            newFlat = []
            
            # Flattening parameters
            flat1 = self.FlattenParams(matingPool[index1])
            flat2 = self.FlattenParams(matingPool[index2])

            #print(flat1)
            
            # Calculating two random crossover points
            pivot1 = np.random.randint(len(flat1) - 2) + 1
            pivot2 = np.random.randint(len(flat1) - 1 - pivot1) + pivot1 + 1

            # Crossover
            newFlat = np.concatenate((flat1[0:pivot1], flat2[pivot1:pivot2]))
            newFlat = np.concatenate((newFlat, flat1[pivot2:len(flat1)]))

            # Mutating the new genome
            newFlat = np.where(np.random.randint(self.mutation_chance, size=(len(newFlat))) == 0,
                               newFlat,
                               np.random.normal(0, newGenome.scale, size=(len(newFlat))))
                               #np.random.rand((len(newFlat)))) * newGenome.scale + newGenome.shift
            
            # Unflattening the genome
            self.UnflattenParams(newFlat, newGenome)

            # Adding the new genome to the population
            self.population.append(newGenome)

        # Adding the top 5% of the previous population
        for idx in range(self.population_size-save_num, self.population_size):
            self.population.append(matingPool[idx - (self.population_size-save_num)])






    # Function that runs the selected number of generations calling the eval_genomes function for every generation
    # It passes the population to the eval_genomes function
    # If the score of a genome or the fitness of some genome reaches the fitness treshold or the score treshold the algorithm saves the best genome to a .pickle file
    # The saving is done after the eval_genomes funcrtion is finished
    # If you wish to save the genome during the run, you can end(return) your eval_genomes function
    # Every time the fitness treshold is reached, delta_fitness will be added to fitness_treshold
    # Every time the score treshold is reached, delta_score will be added to score_treshold
    # Maximum value that the fitness_treshold can reach is max_fitness treshold, after that it will constantly be set to max_fitness_treshold
    # Maximum value that the fitness_score can reach is max_score treshold, after that it will constantly be set to max_score_treshold
    # Your eval_genomes function should return True if you wish to save the best genome of the current population
    # savefile_prefix will be the first part of all files saved by the algorithm
    # If you set save_checkpoints to True whenever the programes saves the best genome, it will save the whole population as well, so you can continue training later from that checkpoint
    # chance of the current parameter being from the first selected parent is 1/crossover_chance
    # chance of the current parameter being set to a random value is 1/mutation_chance
    def run(self, eval_genomes, max_generations, crossover_chance = 2, mutation_chance = 10, fitness_treshold = 1000000, score_treshold = 1000000, delta_fitness = 0, delta_score = 0, max_fitness_treshold = 2000000, max_score_treshold = 2000000, savefile_prefix = '', save_checkpoints = False):
        self.crossover_chance = crossover_chance
        self.mutation_chance = mutation_chance
        self.generationIndex = 1
        # Running for each generation
        for self.generationIndex in range(1, max_generations + 1):
            # Calling the fitness function to get the fitness for each genome
            value = eval_genomes(self.population)
            # Sorting the population by fitness in decending order
            self.population.sort(key = self.compareFunction, reverse = True)
            print(self.generationIndex, ' ', self.population[0].score)
            # Checking if we should save the whole population
            if (value) or (self.population[0].fitness >= fitness_treshold) or (self.population[0].score >= score_treshold):
                # Creating a name for the savefile containg the current generation
                savefile_name = savefile_prefix + 'population-gen' + str(self.generationIndex) + '.pickle'
                # Saving the genome object using the pickle tool
                with open(savefile_name, "wb") as f:
                    pickle.dump(self.population, f)
            # Checking if the user has manually ended the fitness function because he wanted to save the best genome of the current population
            if value:
                # Creating a name for the savefile containing the current fitness_treshold value, the generation index and an indicator that the fitness treshold was reached
                savefile_name = savefile_prefix + 'manual-best-gen' + str(self.generationIndex) + '.pickle'
                # Saving the genome object using the pickle tool
                with open(savefile_name, "wb") as f:
                    pickle.dump(self.population[0], f)
            # Checking if the fitness treshold is reached by the best genome of the population
            if self.population[0].fitness >= fitness_treshold:
                # Creating a name for the savefile containing the current fitness_treshold value, the generation index and an indicator that the fitness treshold was reached
                savefile_name = savefile_prefix + 'fitness' + str(fitness_treshold) + '-best-gen' + str(self.generationIndex) + '.pickle'
                # Saving the genome object using the pickle tool
                with open(savefile_name, "wb") as f:
                    pickle.dump(self.population[0], f)
                # Updating the fitness_treshold value
                fitness_treshold += delta_fitness
                # Checking if the fitness_treshold has reached max_fitness_treshold
                if fitness_treshold > max_fitness_treshold:
                    fitness_treshold = max_fitness_treshold
            # Checking if the score treshold is reached by the best genome of the population
            if self.population[0].score >= score_treshold: 
                # Creating a name for the savefile containing the current score_treshold value, the generation index and an indicator that the score treshold was reached
                savefile_name = savefile_prefix + 'score' + str(score_treshold) + '-best-gen' + str(self.generationIndex) + '.pickle'
                # Saving the genome object using the pickle tool
                with open(savefile_name, "wb") as f:
                    pickle.dump(self.population[0], f)
                # Updating the score_treshold value
                score_treshold += delta_score
                # Checking if the score_treshold has reached max_score_treshold
                if score_treshold > max_score_treshold:
                    score_treshold = max_score_treshold

            # Calling the merging function to get a new generation of the population
            # The new population is stored in self.population

            self.MatingPool10percentRandomIndexMergeSaveTop5percent()
            #self.MatingPool10percentRandomTwoPointCrossoverMergeSaveTop5percent()
            
            '''
            # Printing the population genomes
            for g in self.population:
                print(g.genome)
            print('##################################################')
            '''
            
    
    # Function that returns the index of the current generation
    def ReturnGenerationIndex(self):
        return self.generationIndex
    
# TO DO funkcija koja nastavlja treniranje od sacuvane populacije
# dodaj da stavi self.generationIndex na ondaj koji je bio kad je populalacija sacuvana

# TO DO ret gen idx: proveri da li radi uopste, jer se run vrti for petlju