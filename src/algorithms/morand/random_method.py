import numpy as np
from .utils import *

class RandomMethod:
  
  def __init__(self, nobjs, lb, ub, dims, func, neval = 100, seed = 0):
    self.nobjs = nobjs
    self.ninit = neval
    self.neval = neval
    self.lb = lb
    self.ub = ub
    self.dims = dims
    self.func = func
    self.seed = seed

    self.samples = [[], []] # samples[0] - X, samples[1] - Objs
    self.curt_pareto_samples = [[], []]
    self.sample_counter = 0
    
    np.random.seed(self.seed)


  def start(self):
    init_points = np.random.uniform(low=self.lb, high=self.ub, size=(self.ninit, self.dims))
    while self.sample_counter < self.neval:
      next_X = init_points[self.sample_counter]
      print(f"next: {next_X[:5]}")
      next_Y = self.func(next_X)
      self.samples[0].append(next_X)
      self.samples[1].append(next_Y)
      
      pareto = fast_non_dominated_sort(self.samples[1])
      pareto_samples = [[], []]
      for p in pareto:
          pareto_samples[0].append(self.samples[0][p])
          pareto_samples[1].append(self.samples[1][p])
      self.curt_pareto_samples = pareto_samples
      
      print("samples: ", self.sample_counter)
      print("Found pareto samples:", len(self.curt_pareto_samples[0]),":", self.curt_pareto_samples[1])
      print("="*32)
      self.sample_counter += 1

    print("Optimization reach end after", self.sample_counter, "samples!")
    print("Found pareto samples:", len(self.curt_pareto_samples[0]))
    print(self.curt_pareto_samples[1])
