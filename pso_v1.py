#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import random

import numpy
import math

import matplotlib.pyplot as plt

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


##########################
#Global variables go here#
##########################

POP = 100
GEN = 400
PMIN = -6
PMAX = 6
K = 20
DIM = 10
plt.figure()
random.seed(1234)
numpy.random.seed(1234)
VISUALIZE = 0

##########################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None, inertia=None, cognitive=None, social=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    part.inertia = numpy.random.uniform(0,1)
    part.cognitive = numpy.random.uniform(0,1)
    part.social = numpy.random.uniform(0,1)
    return part

def individual_generate(part):
    indi = creator.Individual([part.inertia,part.cognitive,part.social])
    indi.fitness = part.fitness
    return indi

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    v_u1 = [x * part.cognitive for x in v_u1]
    v_u2 = [x * part.social for x in v_u2]
    part.speed = [x * part.inertia for x in part.speed]
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))

def evolution(ga_pop):
    CXPB, MUTPB = 0.5, .2
    pop = ga_pop
    # Select the next generation individuals
    offspring = toolbox.selectBest(pop)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
    pop[:] = offspring

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)

    return pop


def recalibrate_particles(pop,ga_pop):
    selected = toolbox.selectWorst(pop)
    for i in range(0,len(selected)):
        selected[i].inertia, selected[i].cognitive, selected[i].social = (ga_pop[i])[0],(ga_pop[i])[1], (ga_pop[i])[2]
    return pop
        
def visualize_pso(pop,label,it):
    x = zip(*pop)[0]
    y = zip(*pop)[1]
    plt.plot(x,y,'.')
    plt.axis([PMIN,PMAX,PMIN,PMAX]) 
    name = 'frame' + label + str(0)*(3-len(it))+ str(it) + '.png'
    plt.savefig(name)
    plt.clf()   
    
def heatmap(fun)
    mu = numpy.linspace(-10,10,100)
    gamma = numpy.linspace(-10,10,100)
	
	
    fun_map = numpy.empty((mu.size, gamma.size))
    for i in range(mu.size):
        for j in range(gamma.size):
            fun_map[i,j] = fun([mu[i], gamma[j]])[0]
    
    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1)#, xlabel='$\\gamma$', ylabel='$\\mu$')
    im = s.imshow(
        fun_map,
        extent=(gamma[0], gamma[-1], mu[0], mu[-1]),
        origin='lower')
    fig.show()
		    

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=DIM, pmin=PMIN, pmax=PMAX, smin=-3, smax=3)
toolbox.register("individual", individual_generate)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", benchmarks.ackley)

# register the crossover operator
toolbox.register("mate", tools.cxUniform, indpb=0.5)
# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.

toolbox.register("selectBest", tools.selBest, k=K)
toolbox.register("selectWorst", tools.selWorst, k=K)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)


def parameterfree_pso():
    pop = toolbox.population(n=POP)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None
    

	
    for g in range(GEN):
        ga_pop = []
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
            ga_pop.append(toolbox.individual(part))
        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream) 
        ga_old = list(map(toolbox.clone, ga_pop))
        ga_pop = evolution(ga_pop)    
        print "CONVERGE?"
       	cv = all([all([(i == 0.2) for i in x]) for x in [map(operator.sub,ga_old[i], ga_pop[i]) for i in range(len(ga_pop))]])
       	print cv
       	if cv:
       		print ga_pop[0]
        pop = recalibrate_particles(pop, ga_pop)
        if VISUALIZE == 1:
            visualize_pso(pop,'_pf-',str(g))

    return pop, logbook, best

def normal_pso():
    pop = toolbox.population(n=POP)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

        if VISUALIZE == 1:
            visualize_pso(pop,'_normal-',str(g))

    return pop, logbook, best

a,b,bestpf = parameterfree_pso()
c,d,bestnorm = normal_pso()

print "PF: Best value found: ", benchmarks.ackley(bestpf)[0]
print "Normal: Best value found: ", benchmarks.ackley(bestnorm)[0]
