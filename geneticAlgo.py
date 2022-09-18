import network
import random
import pickle

class Genome:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = 0
        self.score = 0

class Population:
    def __init__(self, population_size, layer_num, architecture):
        self.population_size = population_size
        self.layer_num = layer_num
        self.architecture = architecture
        # Creating a population of the population_size with random genomes with fixed architecutres
        self.population = []
        for i in range(0, self.population_size):
            self.population.append(Genome(network.GetRandomGenomeFixedArchitecture(layer_num, architecture)))
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
    def MatingPool10percentRandomIndexMerge(self):
        # Creating a mating pool with the best 10% of the genomes
        matingPool = []
        for i in range(0, self.population_size//10):
            matingPool.append(self.population[i])
        self.population.clear()

        ''' # Printing the mating pool
        for g in matingPool:
            print(g.genome)
        print('MatingPool length: ', len(matingPool))
        '''

        # Finding random genomes from the mating pool to crossover and mutate the resulting genome
        for _ in range(0, self.population_size):
            # Picking the random genomes
            index1 = random.randrange(0, len(matingPool))
            index2 = random.randrange(0, len(matingPool))

            # Creating and filling the new genome
            newGenome = []
            # Adding the layer_number component of the genome
            newGenome.append(matingPool[index1].genome[0])
            # Adding the fixed architecture of the genome
            for idx in range(1, matingPool[index1].genome[0]+1):
                newGenome.append(matingPool[index1].genome[idx])
            # Adding random values from one of the two selected genomes from the mating pool
            for idx in range(matingPool[index1].genome[0]+1, len(matingPool[index1].genome)):
                if random.randrange(1, 3) == 1:
                    newGenome.append(matingPool[index1].genome[idx])
                else:
                    newGenome.append(matingPool[index2].genome[idx])
            
            # Mutating the new genome
            if random.randrange(1, 3) == 1:
                for _ in range(0, max(1, len(matingPool[index1].genome)//10)+1):
                    newGenome[random.randrange(matingPool[index1].genome[0]+1, len(matingPool[index1].genome))] = random.randrange(-30, 30)
            
            # Adding the new genome to the population
            self.population.append(Genome(newGenome))

    # This function works the same as the one before, but it saves the top max(1, 5%) of the old population into the new one
    def MatingPool10percentRandomIndexMergeSaveTop5percent(self):
        # Creating a mating pool with the best 10% of the genomes
        matingPool = []
        for i in range(0, self.population_size//10):
            matingPool.append(self.population[i])
        self.population.clear()

    
        '''
        # Printing the mating pool
        for g in matingPool:
            print(g.genome)
        print('MatingPool length: ', len(matingPool))
        '''
        

        # Calculating the number of genomes that will be saved from the currnet population
        save_num = max(self.population_size//20, 1)

        # Finding random genomes from the mating pool to crossover and mutate the resulting genome
        for _ in range(0, self.population_size-save_num):
            # Picking the random genomes
            index1 = random.randrange(0, len(matingPool))
            index2 = random.randrange(0, len(matingPool))

            # Creating and filling the new genome
            newGenome = []
            # Adding the layer_number component of the genome
            newGenome.append(matingPool[index1].genome[0])
            # Adding the fixed architecture of the genome
            for idx in range(1, matingPool[index1].genome[0]+1):
                newGenome.append(matingPool[index1].genome[idx])
            # Adding random values from one of the two selected genomes from the mating pool
            for idx in range(matingPool[index1].genome[0]+1, len(matingPool[index1].genome)):
                if random.randrange(1, 3) == 1:
                    newGenome.append(matingPool[index1].genome[idx])
                else:
                    newGenome.append(matingPool[index2].genome[idx])
            # Mutating the new genome
            if random.randrange(1, 3) == 1:
                for _ in range(0, max(1, len(matingPool[index1].genome)//10)+1):
                    newGenome[random.randrange(matingPool[index1].genome[0]+1, len(matingPool[index1].genome))] = random.randrange(-30, 30)
            
            # Adding the new genome to the population
            self.population.append(Genome(newGenome))

        # Adding the top 5% of the previous population
        for idx in range(self.population_size-save_num, self.population_size):
            self.population.append(matingPool[idx - (self.population_size-save_num)])


    # Function that takes the best 10 percent of a population and puts it into a mating pool
    # It selects random genomes from the pool and goes through the genome and creates a new genome
    # It selects a random index, the first part of the new genome will be filled with values from the genome at index1
    # The second part of the new genome will be filled with the values from the genome at index2
    # The architecture of the network won't change
    # There is a 50% chance for each genome to be mutated
    # If a genome is chosen to be mutated, max(1, 10%) of its values will be mutated, architecture won't change
    # The % of changed values may be smaller because the same value can be chosen to mutate multiple times
    # This function saves the top max(1, 5%) of the old population into the new one
    def MatingPoolRandomIndexCrossoverSaveTop5percent(self):
        # Creating a mating pool with the best 10% of the genomes
        matingPool = []
        for i in range(0, self.population_size//10):
            matingPool.append(self.population[i])
        self.population.clear()
    
        '''
        # Printing the mating pool
        for g in matingPool:
            print(g.genome)
        print('MatingPool length: ', len(matingPool))
        '''

        # Calculating the number of genomes that will be saved from the currnet population
        save_num = max(self.population_size//20, 1)

        # Finding random genomes from the mating pool to crossover and mutate the resulting genome
        for _ in range(0, self.population_size-save_num):
            # Picking the random genomes
            index1 = random.randrange(0, len(matingPool))
            index2 = random.randrange(0, len(matingPool))

            # Creating and filling the new genome
            newGenome = []
            # Adding the layer_number component of the genome
            newGenome.append(matingPool[index1].genome[0])
            # Adding the fixed architecture of the genome
            for idx in range(1, matingPool[index1].genome[0]+1):
                newGenome.append(matingPool[index1].genome[idx])
            # Randomly finding the point at which will the genes crossover
            crossover_index = random.randrange(matingPool[index1].genome[0]+1, len(matingPool[index1].genome)-2)
            # Filling the first part of the genome with the values from index1
            for idx in range(matingPool[index1].genome[0]+1, crossover_index+1):
                newGenome.append(matingPool[index1].genome[idx])
            # Filling the second part of the genome with the values from index2
            for idx in range(crossover_index+1, len(matingPool[index1].genome)):
                newGenome.append(matingPool[index2].genome[idx])

            # Mutating the new genome
            if random.randrange(1, 3) == 1:
                for _ in range(0, max(1, len(matingPool[index1].genome)//10)+1):
                    newGenome[random.randrange(matingPool[index1].genome[0]+1, len(matingPool[index1].genome))] = random.randrange(-30, 30)
            
            # Adding the new genome to the population
            self.population.append(Genome(newGenome))

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
    def run(self, eval_genomes, max_generations, fitness_treshold = 1000000, score_treshold = 1000000, delta_fitness = 0, delta_score = 0, max_fitness_treshold = 2000000, max_score_treshold = 2000000, savefile_prefix = '', save_checkpoints = False):
        self.generationIndex = 1
        # Running for each generation
        for self.generationIndex in range(1, max_generations + 1):
            # Calling the fitness function to get the fitness for each genome
            value = eval_genomes(self.population)
            # Sorting the population by fitness in decending order
            self.population.sort(key = self.compareFunction, reverse = True)
            print(self.generationIndex, ' ', self.population[0].fitness)
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
                    pickle.dump(self.population[0].genome, f)
            # Checking if the fitness treshold is reached by the best genome of the population
            if self.population[0].fitness >= fitness_treshold:
                # Creating a name for the savefile containing the current fitness_treshold value, the generation index and an indicator that the fitness treshold was reached
                savefile_name = savefile_prefix + 'fitness' + str(fitness_treshold) + '-best-gen' + str(self.generationIndex) + '.pickle'
                # Saving the genome object using the pickle tool
                with open(savefile_name, "wb") as f:
                    pickle.dump(self.population[0].genome, f)
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
                    pickle.dump(self.population[0].genome, f)
                # Updating the score_treshold value
                score_treshold += delta_score
                # Checking if the score_treshold has reached max_score_treshold
                if score_treshold > max_score_treshold:
                    score_treshold = max_score_treshold

            # Calling the merging function to get a new generation of the population
            # The new population is stored in self.population
            self.MatingPool10percentRandomIndexMergeSaveTop5percent()

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