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

import os
import glob
import subprocess

import matplotlib.pyplot as plt

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


##########################
#Global variables go here#
##########################

POP = 50
GEN = 50
PMIN = -32
PMAX = 32
K = 5                   # 5, 10, 25, 50
DIM = 50
plt.figure()
random.seed(1234)
numpy.random.seed(1234)
ilb = 0.75              # inertia_lower_bound, should be in [0,1], step length .1

GROWING_PARAM = 1       # Parameters grow during the course of the algo
growth_rate = 1.001     # at each iteration, parameters will bu multiplied by this rate
ALTERNATE_GA_FIT = 0    # Use alternative fitness for ga, 0,1,2
fit_weight = 0.5        # [0, 2], step .5
bestfit_weight = 1.0

VISUALIZE = 0
VISUALIZE_PARAM = 0
VISUALIZE_GRAPHS = 0
heatmap_threshold = 0   # if non zero, heatmap values greater than the threshold will not be displayed
ANIMATE = 0

##########################

def set_parameters(k, ilb, growing_param, alternate_ga_fit, fit_weight):
    K = k
    ilb = ilb
    GROWING_PARAM = growing_param
    ALTERNATE_GA_FIT = alternate_ga_fit
    fit_weight = fit_weight

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None, inertia=None, cognitive=None, social=None,
    lastFit=None)
    
def whitley(x):
    """
    Whitley test function
    Usually used in [-10.24, 10.24]^n
    Global optimum is 0 for [1,1,...,1]
    """

    fitness = 0
    limit = len(x)
    for i in range(limit):
        for j in range(limit):
            temp = 100*((x[i]**2)-x[j]) + \
                (1-x[j])**2
            fitness += (float(temp**2)/4000.0) - math.cos(temp) + 1
    return fitness,


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    part.inertia = numpy.random.uniform(ilb,1)
    part.cognitive = numpy.random.uniform(0,1)
    part.social = numpy.random.uniform(0,1)
    return part



def individual_generate(part):
    indi = creator.Individual([part.inertia,part.cognitive,part.social])
    if ALTERNATE_GA_FIT == 1:
        fit, lastfit = part.fitness.values[0],part.lastFit[0]
        improvement = max(0,fit-lastfit)
        indi.fitness = (fit/(improvement+1),)
    elif ALTERNATE_GA_FIT == 2:
        fit, bestfit = part.fitness.values[0],part.best.fitness.values[0]
        newfit = fit_weight*fit + bestfit_weight*bestfit
        indi.fitness = (newfit,)
    else :
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
    CXPB, MUTPB = 0.5, .3
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

def diversity(pop):
    n = len(pop)
    d = 0
    for i in range(n):
        for j in range(i):
            for x, y in zip(pop[i], pop[j]):
                d += (x - y)**2
    return 2*d/(n*(n-1))

def recalibrate_particles(pop,ga_pop):
    selected = toolbox.selectWorst(pop)
    for i in range(0,len(selected)):
        selected[i].inertia, selected[i].cognitive, selected[i].social = (ga_pop[i])[0],(ga_pop[i])[1], (ga_pop[i])[2]
    return pop
        
def visualize_pso(pop,label,it,heatmap):
    plt.imshow(heatmap,extent=(PMIN,PMAX,PMIN,PMAX), origin ='lower')
    plt.colorbar()


    x = zip(*pop)[0]
    y = zip(*pop)[1]
    lastx = [part[0] - part.speed[0] for part in pop]
    lasty = [part[1] - part.speed[1] for part in pop]
    plt.plot([x,lastx],[y,lasty],'k',linewidth=0.2)
    plt.plot(x,y,'k.')
    plt.axis([PMIN,PMAX,PMIN,PMAX]) 
    name = 'frame' + label + str(0)*(3-len(it))+ str(it) + '.png'
    plt.savefig(name)
    plt.clf()

#    plt.show()

def visualize_params(pop, it):
    inert = [part.inertia for part in pop]
    cog = [part.cognitive for part in pop]
    soc = [part.social for part in pop]
    plt.hist(inert, bins = numpy.linspace(0,2,40), alpha = .33, label='Inertia')
    plt.hist(cog, bins = numpy.linspace(0,2,40), alpha = .33, label='Cognitive')
    plt.hist(soc, bins = numpy.linspace(0,2,40), alpha = .33, label='Social')
    plt.xlim([0,2])
    plt.ylim([0,POP])
    name = 'param' + str(0)*(3-len(str(it)))+ str(it) + '.png'
    plt.legend()
    plt.savefig(name)
    plt.clf()
    
def plotgraph(logbook, pf_flag):
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["Fitness"].select("min")
    div = logbook.chapters["Population"].select("div")

    gen, fit_mins,div = gen[50:], fit_mins[50:], [math.log(x,10) for x in div[50:]]

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, div, "r-", label="Population Diversity")
    ax2.set_ylabel("Diversity", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    if pf_flag == 1:
        plt.savefig("pf_graph.png")
    else:
        plt.savefig("norm_graph.png")
    plt.clf()

    if pf_flag == 1:
        gen = logbook.select("gen")
        ine = logbook.chapters["Inertia"].select("avg")
        soc = logbook.chapters["Social"].select("avg")
        cog = logbook.chapters["Cognitive"].select("avg")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, ine, "b-", label="Inertia")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        line2 = ax1.plot(gen, soc, "g-", label="Social")
        for tl in ax1.get_yticklabels():
            tl.set_color("g")

        line3 = ax1.plot(gen, cog, "r-", label="Cognitive")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Average value", color="k")
        for tl in ax1.get_yticklabels():
            tl.set_color("r")


        lns = line1 + line2 + line3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="upper left")

        plt.savefig("params_graph.png")
        
def createheatmap(fun):
    mu = numpy.linspace(PMIN,PMAX,100)
    gamma = numpy.linspace(PMIN,PMAX,100)
	
	
    fun_map = numpy.empty((mu.size, gamma.size))
    for i in range(mu.size):
        for j in range(gamma.size):
            fun_map[i,j] = fun([mu[i], gamma[j]])[0]
            
    if heatmap_threshold > 0:   
        hi_indices = fun_map > heatmap_threshold
        fun_map[hi_indices] = heatmap_threshold
    
    return fun_map
		    

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=DIM, pmin=PMIN, pmax=PMAX, smin=(PMIN-PMAX)/10., smax=(PMAX-PMIN)/10.)
toolbox.register("individual", individual_generate)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", benchmarks.ackley)

# register the crossover operator
toolbox.register("mate", tools.cxBlend, alpha=0.0)
# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.

toolbox.register("selectBest", tools.selBest, k=K)
toolbox.register("selectWorst", tools.selWorst, k=K)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)


def parameterfree_pso():
    pop = toolbox.population(n=POP)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_pop = tools.Statistics()
    stats_ine = tools.Statistics(lambda ind: ind.inertia)
    stats_soc = tools.Statistics(lambda ind: ind.social)
    stats_cog = tools.Statistics(lambda ind: ind.cognitive)
    stats = tools.MultiStatistics(Fitness = stats_fit, Population = stats_pop, Inertia = stats_ine, Social = stats_soc, Cognitive = stats_cog)
    stats_fit.register("avg", numpy.mean)
    stats_fit.register("std", numpy.std)
    stats_fit.register("min", numpy.min)
    stats_fit.register("max", numpy.max)
    stats_ine.register("avg", numpy.mean)
    stats_ine.register("std", numpy.std)
    stats_soc.register("avg", numpy.mean)
    stats_soc.register("std", numpy.std)
    stats_cog.register("avg", numpy.mean)
    stats_cog.register("std", numpy.std)
    stats_pop.register("div", diversity)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "Fitness", "Inertia", "Social", "Cognitive", "Population"]

    best = None
    
    if VISUALIZE == 1:
        heatmap = createheatmap(toolbox.evaluate)
    for part in pop:
        part.fitness.values = toolbox.evaluate(part)

	
    for g in range(GEN):
        if VISUALIZE_PARAM == 1:
            visualize_params(pop,g)
        
        ga_pop = []
        for part in pop:
            part.lastFit = part.fitness.values
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            if GROWING_PARAM == 1:
                part.inertia   = min(part.inertia*growth_rate,2)
                part.cognitive = min(part.cognitive*growth_rate,2)
                part.social    = min(part.social*growth_rate,2)
            toolbox.update(part, best)
            ga_pop.append(toolbox.individual(part))
        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream) 
#        ga_old = list(map(toolbox.clone, ga_pop))
        ga_pop = evolution(ga_pop)    
#        print "CONVERGE?"
#       	cv = all([all([(i < 0.2) for i in x]) for x in [map(operator.sub,ga_old[i], ga_pop[i]) for i in range(len(ga_pop))]])
#       	print cv
#       	if cv:
#       		print ga_pop[0]
        pop = recalibrate_particles(pop, ga_pop)
        if VISUALIZE == 1:
            visualize_pso(pop,'_pf-',str(g),heatmap)

    if VISUALIZE_GRAPHS == 1:
        plotgraph(logbook,1)

    return pop, logbook, best

def normal_pso():
    pop = toolbox.population(n=POP)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_pop = tools.Statistics()
    stats = tools.MultiStatistics(Fitness = stats_fit, Population = stats_pop)
    stats_fit.register("avg", numpy.mean)
    stats_fit.register("std", numpy.std)
    stats_fit.register("min", numpy.min)
    stats_fit.register("max", numpy.max)
    stats_pop.register("div", diversity)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

#    for part in pop:
#        part.inertia, part.social, part.cognitive = 1,1,1

    if VISUALIZE == 1:
        heatmap = createheatmap(toolbox.evaluate)

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
            visualize_pso(pop,'_normal-',str(g),heatmap)

    if VISUALIZE_GRAPHS == 1:
        plotgraph(logbook,0)

    return pop, logbook, best

i = 0
test_val_k = [5, 10, 25, 50]
test_val_ilb = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
test_val_growing_param = [0, 1]
test_val_alternate_ga_fit = [0, 1, 2]
test_val_fit_weight = [0, .5, 1, 1.5, 2]

total_len = len(test_val_k)+len(test_val_ilb)+len(test_val_growing_param)+len(test_val_alternate_ga_fit)+len(test_val_fit_weight)

while True:
    
#    f.write("-------------------\n")
#    f.write("\nITERATION "+str(i)+":\n")
#    f.write("-------------------\n")
   
    for test_k in test_val_k:
        for test_ilb in test_val_ilb:
            for test_growing_param in test_val_growing_param:
                for test_alternate_ga_fit in test_val_alternate_ga_fit:
                    for test_fit_weight in test_val_fit_weight:

                        set_parameters(test_k, test_ilb, test_growing_param, test_alternate_ga_fit, test_fit_weight)

                        f = open("output/output-"+str(test_k)+"-"+str(test_ilb)+"-"+str(test_growing_param)+"-"+str(test_alternate_ga_fit)+"-"+str(test_fit_weight)+".txt", "a")

                        a,b,bestpf = parameterfree_pso()
                        #c,d,bestnorm = normal_pso()

                        best_pf = toolbox.evaluate(bestpf)
                        #best_norm = toolbox.evaluate(bestnorm)


                        f.write(str(test_k)+"\n")
                        f.write(str(test_ilb)+"\n")
                        f.write(str(test_growing_param)+"\n")
                        f.write(str(test_alternate_ga_fit)+"\n")
                        f.write(str(test_fit_weight)+"\n")
                        f.write(str(best_pf)+"\n")
                        f.write("\n")

    f.close()

    i+=1

if VISUALIZE == 1 and ANIMATE == 1:

    str = "Generating animation."
    print str

    for filename in glob.glob("frame_normal*.png"):

        str += "."
        print str

        number = filename[12:16]
        subprocess.call("convert frame_normal"+number+".png frame_pf"+number+".png +append frame_out"+number+".png", shell=True)

    subprocess.call("convert -delay 10 -loop 0 frame_out*.png final.gif", shell=True)

    for filename in glob.glob("frame_*.png"):
        os.remove(filename)

    print "Animation done, see final.gif"


print "PF: Best value found: ", toolbox.evaluate(bestpf)
print "Normal: Best value found: ", toolbox.evaluate(bestnorm)


