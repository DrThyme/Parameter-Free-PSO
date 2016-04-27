from pso import pso
from parafreePSO import pso as pfpso
import math

def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    
def ack(x):
    firstSum = 0.0
    secondSum = 0.0
    for c in x:
        firstSum += c**2.0
        secondSum += math.cos(2.0*math.pi*c)
    n = float(len(x))
    return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e
    
def sphere(x):
    return sum(x**2)
    
    
def rostrigin(x):
	psum = 0.0
	for i in x:
		psum = (x[i]**2-10*math.cos(2*math.pi*x[i])+10)
	return psum
	
	
def griewank(x):
	part1 = 0
	for i in range(len(x)):
		part1 += x[i]**2
		part2 = 1
	for i in range(len(x)):
		part2 *= math.cos(float(x[i]) / math.sqrt(i+1))
	return 1 + (float(part1)/4000.0) - float(part2)
	
	
	
def withley(x):
	fitness = 0
	limit = len(x)
	for i in range(limit):
		for j in range(limit):
			temp = 100*((x[i]**2)-x[j]) + \
				(1-x[j])**2
			fitness += (float(temp**2)/4000.0) - math.cos(temp) + 1
	return fitness
	
	
def con(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]

lb = [-10, -10, -10, -10, -10, -10]
ub = [10, 10, 10, 10, 10, 10]


print("###################################################")
print("Normal PSO")
print("###################################################")
print pso(withley, lb, ub, particle_output=False)
print("###################################################")
print("Parameter Free PSO")
print("###################################################")
print pfpso(withley, lb, ub, particle_output=False)
print("###################################################")
