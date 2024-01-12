import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .utils import *
import warnings
warnings.filterwarnings("ignore")

noise        =   0.1
m52          =   ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

class MOBO:

  def __init__(self, nobjs, ninit, lb, ub, dims, func, neval=100, seed=42):
    self.nobjs = nobjs
    self.ninit = ninit
    self.neval = neval
    self.lb = lb
    self.ub = ub
    self.dims = dims
    self.func = func
    self.seed = seed

    self.samples = [[], []] # samples[0] - X, samples[1] - Objs
    self.curt_pareto_samples = [[], []]
    self.sample_counter = 0
    self.negative = 1
    
    optimal="min"
    if optimal == "min":
      self.negative = -1

    self.gprs = [GaussianProcessRegressor(kernel=m52, alpha=noise**2) for i in range(nobjs)]

    np.random.seed(self.seed)

  def start(self):

    self.init_train()
    print(f"After init, the sample size is: {len(self.samples[0])}")
    while self.sample_counter < self.neval:
      next_X = self.select()
      self.collect(next_X)
      print("Found pareto samples:", len(self.curt_pareto_samples[0]),":", self.curt_pareto_samples[1])
      print("Overall samples:", len(self.samples[0]))
      print("="*32, flush=True)
      self.train_gpr()
    print("Optimization reach end after", self.sample_counter, "samples!")
    print("Found pareto samples:", len(self.curt_pareto_samples[0]))
    print(self.curt_pareto_samples[1])

  def init_train(self):
    init_points = latin_hypercube(self.ninit, self.dims)
    init_points = from_unit_cube(init_points, self.lb, self.ub)
    count = 0
    for point in init_points:
      print(f"begin to init point {count + 1}", flush=True)
      self.collect(point)
      count += 1
    self.train_gpr()
  
  def train_gpr(self):

    X = np.asarray(self.samples[0]).reshape(-1,self.dims)
    objs = np.asarray(self.samples[1]).T
    # print("Training GPRs with {} samples.".format(len(self.samples[0])))
    for i in range(self.nobjs):
      # some sort of reashape here
      self.gprs[i].fit(X, objs[i])
      

  def collect(self, X):

    if self.sample_counter >= self.neval:
      return

    objs = self.func(X)
    print(f"collect X: {X[:5]}, objs: {objs}")
    nega_objs = [o * self.negative for o in objs]
    self.samples[0].append(X)
    self.samples[1].append(nega_objs)
    pareto = fast_non_dominated_sort(self.samples[1])
    pareto_samples = [[], []]

    for p in pareto:
      pareto_samples[0].append(self.samples[0][p])
      pareto_objs = self.samples[1][p]
      nega_objs = [o * self.negative for o in pareto_objs]
      pareto_samples[1].append(nega_objs)
    self.curt_pareto_samples = pareto_samples
    self.sample_counter += 1
  
  '''
    @input: X - np.ndarray, shape = (n, dims)
    @output: np.ndarray, shape = (n, nobjs)
  '''
  def ucb(self, X: np.ndarray, kappa = 2.0) -> np.ndarray:
    mean, std = [], []
    for i in range(self.nobjs):
      m, s = self.gprs[i].predict(X, return_std=True)
      mean.append(m)
      std.append(s)
    mean = np.asarray(mean).T
    std = np.asarray(std).T
    return mean + kappa * std

  def select(self):
    Xs = np.random.uniform(self.lb, self.ub, (1000, self.dims))
    Ys = self.ucb(Xs) 
    Ys_list = Ys.tolist()
    pareto_idx = fast_non_dominated_sort(Ys_list)
    # choose a random index from pareto_idx
    idx = np.random.choice(pareto_idx)
    print(f"choose index: {idx}")
    return Xs[idx]