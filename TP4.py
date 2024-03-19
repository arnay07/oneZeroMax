import random
from math import sqrt, exp, pow, sin
#importation de problem 
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.problem import OneMax
from random import randint 

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization.plotting import Plot
from jmetal.util.solution import *
from jmetal.core.quality_indicator import *


#definition du nouveau probleme MO OneZeroMax
# 
class OneZeroMax(BinaryProblem):
    def __init__(self, number_of_bits):
        super(OneZeroMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 2
        self.number_of_variables = 1
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE,self.MINIMIZE]
        self.obj_labels = ['Ones','Zero']

    def number_of_variables(self) -> int:
        return 1

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0 

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        counter_of_ones = 0
        counter_of_zeroes = 0
        for bits in solution.variables[0]:
            if bits:
                counter_of_ones += 1
            else:
                counter_of_zeroes += 1

        solution.objectives[0] = -1.0 * counter_of_ones
        solution.objectives[1] = -1.0 * counter_of_zeroes

        return solution

    def create_solution(self):
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=2)
        new_solution.variables[0] = \
        [True if randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def name(self) -> str:
        return 'OneZeroMax'


problem = OneZeroMax(256)

max_evaluations = 25000

# On definit l'algo NSGA II
algorithm = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=BitFlipMutation(1.0 / problem.number_of_bits),
    crossover=SPXCrossover(0.9),
    termination_criterion=StoppingByEvaluations(max_evaluations)
)

# On le fait tourner
algorithm.run()
solutions = algorithm.get_result()

# On définit d'autres algorithmes 

#On trace le front pareto

front = get_non_dominated_solutions(solutions)
plot_front = Plot('Pareto front approximation', axis_labels=['ones', 'zeros'])
plot_front.plot(front, label='NSGAII-OneZeroMax')


# Save results to file
print_function_values_to_file(front, "FUN." + algorithm.label)
print_variables_to_file(front, "VAR." + algorithm.label)

print(f"Algorithm: {algorithm.get_name()}")
print(f"Problem: {problem.name()}")
print(f"Computing time: {algorithm.total_computing_time}")

# Calcul de métriques
#HyperVolume(reference_point=[0, 0]).compute(front)
indicator = HyperVolume([0, 0])
objectives = []
for solution in solutions:
    objectives.append(solution.objectives)
    hv = round(indicator.compute(objectives), 0)
print(f"Hypervolume: ", str(hv))