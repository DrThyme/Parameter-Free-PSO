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

POP = 20
GEN = 500
PMIN = -5
PMAX = 5
K = 5
DIM = 20
plt.figure()
random.seed(1234)
numpy.random.seed(1234)
ilb = 0.75              # inertia_lower_bound, should be in [0,1]

GROWING_PARAM = 0       # Parameters grow during the course of the algo
growth_rate = 1.001     # at each iteration, parameters will bu multiplied by this rate
ALTERNATE_GA_FIT = 0    # Use alternative fitness for ga
fit_weight = 0.5
bestfit_weight = 1.0

VISUALIZE = 0
VISUALIZE_PARAM = 0
VISUALIZE_GRAPHS = 1
heatmap_threshold = 0   # if non zero, heatmap values greater than the threshold will not be displayed
ANIMATE = 0

##########################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None, inertia=None, cognitive=None, social=None,
    lastFit=None, prevFit = None)
    
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
    
    
def popstats(pop):
    n = len(pop)
    dist = [[0 for x in range(n-1)] for y in range(n)]
    for i in range(n):
        for j in range(i):
            d = 0
            for x, y in zip(pop[i], pop[j]):
                d += (x - y)**2
            d = math.sqrt(d)
            dist[i][j] = d
            dist[j][i-1] = d
    meandist = [numpy.mean(x) for x in dist]
    maxdist = [numpy.max(x) for x in dist]
    mindist = [numpy.min(x) for x in dist]
    minmin = numpy.min(mindist)
    meanmin = numpy.mean(mindist)
    maxmin = numpy.max(mindist)
    minmean = numpy.min(meandist)
    meanmean = numpy.mean(meandist)
    maxmean = numpy.max(meandist)
    minmax = numpy.min(maxdist)
    meanmax = numpy.mean(maxdist)
    maxmax = numpy.max(maxdist)
    c1 = len([0 for x in maxdist if x < minmin])
    c2 = len([0 for x in maxdist if x < meanmin])
    c3 = len([0 for x in maxdist if x < maxmin])
    c4 = len([0 for x in maxdist if x < minmean])
    c5 = len([0 for x in maxdist if x < meanmean])
    c6 = len([0 for x in maxdist if x < maxmean])
    c7 = len([0 for x in maxdist if x < minmax])
    c8 = len([0 for x in maxdist if x < meanmax])
    c9 = len([0 for x in maxdist if x < maxmax])
    starvecrit =  numpy.mean([len([0 for x in y if x < minmean]) for y in dist]) #average nbof neighbour closerthan minimum average distance
    starvevec = [len([0 for x in y if x < minmean]) for y in dist]
    starving = len([0 for x in starvevec if x > 0.4*POP])
    isolcrit =  numpy.mean([len([0 for x in y if x < maxmean]) for y in dist])
    isolvec = [len([0 for x in y if x < maxmean]) for y in dist]
    isolated = len([0 for x in isolvec if x < 0.6*POP])
    healthcrit =  numpy.mean([len([0 for x in y if x < meanmean]) for y in dist])
    healthvec = [len([0 for x in y if x < meanmean]) for y in dist]
#    print healthvec
    healthy = len([0 for x in healthvec if x > 0.4*POP if x < 0.6*POP])
#    print [c1,c2,c3,c4,c5,c6,c7,c8,c9]
    return [starving, healthy, isolated, starving+healthy+isolated]
    
    
    
def update_flags(pop):
    n = len(pop)
    dist = [[0 for x in range(n-1)] for y in range(n)]
    for i in range(n):
        for j in range(i):
            d = 0
            for x, y in zip(pop[i], pop[j]):
                d += (x - y)**2
            d = math.sqrt(d)
            dist[i][j] = d
            dist[j][i-1] = d
    meandist = [numpy.mean(x) for x in dist]
    maxdist = [numpy.max(x) for x in dist]
    mindist = [numpy.min(x) for x in dist]
    meanmin = numpy.mean(mindist)
    maxmin = numpy.max(mindist)
    minmean = numpy.min(meandist)
    meanmean = numpy.mean(meandist)
    maxmean = numpy.max(meandist)
    meanmax = numpy.mean(maxdist)
    minmax = numpy.min(maxdist)
    
    lbpop = 0.4*n
    ubpop = 0.6*n
    
    starvecrit =  numpy.mean([len([0 for x in y if x < minmean]) for y in dist])
    starvevec = [len([0 for x in y if x < minmean]) for y in dist]
    starveprob = [(x-lbpop)/ubpop for x in starvevec]
    
    isolcrit =  numpy.mean([len([0 for x in y if x < maxmean]) for y in dist])
    isolvec = [len([0 for x in y if x < maxmean]) for y in dist]
    isolprob = [(x)/lbpop for x in starvevec]
    
    healthcrit =  numpy.mean([len([0 for x in y if x < meanmean]) for y in dist])
    healthvec = [len([0 for x in y if x < meanmean]) for y in dist]
    breeders = [1 if (x > lbpop and x < ubpop) else 0 for x in healthvec]
    
    markedforremoval = [0 if ((1-numpy.random.exponential(.01)) < starveprob[i] or (numpy.random.exponential(.01)) > isolprob[i]) else 1 for i in range(n)]
    removeflag = numpy.nonzero(markedforremoval)[0]
    breedflag = numpy.nonzero(breeders)[0]
    print sum(markedforremoval)
    return removeflag, breeders, sum(breeders)
    
    
    
    

def breed_particles(pop,ga_pop):
    children = []
    for i in range(0,len(pop)):
        child = creator.Particle(pop[i].best)
        child.speed = [random.uniform(pop[i].smin, pop[i].smax) for _ in range(DIM)]
        child.smin = pop[i].smin
        child.smax = pop[i].smax
        child.inertia = (ga_pop[i])[0]
        child.cognitive = (ga_pop[i])[1]
        child.social = (ga_pop[i])[2]
        child.fitness.values = pop[i].fitness.values
        children.append(child)
    return children
        
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
toolbox.register("evaluate", benchmarks.schaffer)

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
    st = []
    if VISUALIZE == 1:
        heatmap = createheatmap(toolbox.evaluate)
    for part in pop:
        part.fitness.values = toolbox.evaluate(part)

	
    for g in range(GEN):
        if VISUALIZE_PARAM == 1:
            visualize_params(pop,g)
        
        ga_pop = []
        for part in pop:
            part.prevFit = part.fitness.values
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
        removeflag, breeders, newpart = update_flags(pop)
        helpflag = [1 if (part.fitness.values > part.lastFit or part.lastFit > part.prevFit) else 0 for part in pop]
        breedflag = [helpflag[i]*breeders[i] for i in range(len(breeders))]
        breedflag = numpy.nonzero(breedflag)[0]
        newpart = len(breedflag)
        print newpart
        toolbox.register("selectBest", tools.selBest, k=newpart)
        ga_pop = evolution(ga_pop)
#        print "CONVERGE?"
#       	cv = all([all([(i < 0.2) for i in x]) for x in [map(operator.sub,ga_old[i], ga_pop[i]) for i in range(len(ga_pop))]])
#       	print cv
#       	if cv:
#       		print ga_pop[0]
        children = breed_particles([pop[i] for i in breedflag], ga_pop)
        pop = [pop[i] for i in removeflag]+children
        st.append(popstats(pop))
        if VISUALIZE == 1:
            visualize_pso(pop,'_pf-',str(g),heatmap)
        

    if VISUALIZE_GRAPHS == 1:
        plotgraph(logbook,1)
        
    minmin = [x[0] for x in st]
    meanmin = [x[1] for x in st]
    maxmin = [x[2] for x in st]
    sumclass = [x[3] for x in st]
    ax = plt.subplot(111)
    ax.plot(minmin, label='starving')
    ax.plot(meanmin, label='healthy')
    ax.plot(maxmin, label='isolated')
    ax.plot(sumclass, label='sum')
        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.clf()

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

a,b,bestpf = parameterfree_pso()
random.seed(1234)
numpy.random.seed(1234)
c,d,bestnorm = normal_pso()

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


