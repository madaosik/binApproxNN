##========== Copyright (c) 2021, Adam Lanicek, All rights reserved. =========##
##
## Purpose:     Module defining the properties & running the NSGA-II algorithm 
##              implemented in the pymoo Python library (https://pymoo.org/).
##
##              The module runs the NSGA-II algorithm with the inputs defined
##              in the optimSetup.py module. The algorithm generates 5 integer array
##              indices in the range of 0 - len(mults)-1 and assigns the 
##              multiplier at that index to one of the 5 FakeConv layers.
##
## Implement.:  Implemented by Adam Lanicek based on the pymoo Getting started guide.
##
## $Date:       $2021-05-05
##============================================================================##

import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.visualization.scatter import Scatter
import optimSetup as setup
import subprocess


class ApproxCNNOptimization(Problem):

    def __init__(self, maxInd, nVar):
        super().__init__(n_var=nVar,
                         n_obj=2,
                         n_constr=0,
                         xl=np.full(nVar,0),
                         xu=np.full(nVar, maxInd),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        acc = self.compute_accuracy(x)
        energy = self.compute_energy(x)

        out["F"] = [energy, -acc]

    def compute_accuracy(self, x):
        """
        Launches the AlexNet evaluation subprocesses with the given multipliers assigned to specific layers

        @param x = [index of multiplier for layer 1, multiplier for layer 2 ......]
        """
        m = [setup.BIN_PATH + setup.PREF + setup.mults[layerInd][setup.MULT_NAME] + setup.SUFF for layerInd in x]
        params = ["python", "eval_AlexNet.py", "--fakeConv", "--m1", m[0], "--m2", m[1], "--m3", m[2], "--m4", m[3], "--m5", m[4]]
        output = subprocess.check_output(params,universal_newlines=True)

        acc = output.split('\n')[-2].split(':')[1].strip()
        return float(acc)


    def compute_energy(self, x):
        """
        Computes the total energy consumed based on the total number of multiplications in convolution layers
        and multipliers energy requirements.

        @param x = [index of multiplier for layer 1, multiplier for layer 2 ......]
        """
        m_en = [setup.mults[layerInd][setup.MULT_EN] for layerInd in x]
        return np.sum(np.multiply(m_en,setup.convLayersMult))/1000

problem = ApproxCNNOptimization(len(setup.mults)-1, len(setup.convLayersMult))

algorithm = get_algorithm("nsga2",
                       pop_size=50,
                       sampling=get_sampling("int_random"),
                       crossover=get_crossover("int_sbx", prob=0.5, eta=3.0),
                       mutation=get_mutation("int_pm", eta=3.0),
                       eliminate_duplicates=True,
                       )

res = minimize(problem,
               algorithm,
               ("n_gen", 50),
               verbose=True,
               save_history=True,
               seed=1)

for result in res.F:
    result[1] = -result[1]*100

print("Best solution found: %s" % res.X)
print("Function value:")
for el in res.F:
    print(list(map('{:.2f}'.format, el)))

plot = Scatter(labels=["Energy requirements (Watts)", "Accuracy (%)"])
plot.add(res.F, color="red")
plot.show()