from pso import pso
import math

def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    
def ack(chromosome):
    firstSum = 0.0
    secondSum = 0.0
    for c in chromosome:
        firstSum += c**2.0
        secondSum += math.cos(2.0*math.pi*c)
    n = float(len(chromosome))
    return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e
	
	
def con(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]

lb = [-32, -32, -32, -32, -32, -32]
ub = [32, 32, 32, 32, 32, 32]

xopt, fopt, p, fp = pso(rosen, lb, ub, particle_output=True)
print xopt, fopt, p, fp
