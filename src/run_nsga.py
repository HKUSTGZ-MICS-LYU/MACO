import os
import time
import glob
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
# import from projects
from design_space import DesignSpace

_simba_dir = "../examples/simba"


class MyProblem(ElementwiseProblem):
  def __init__(self, workload_dir: str, tmp_output_dir: str):
    arch_files_wildcard = [_simba_dir + "/arch/*.yaml", _simba_dir + "/arch/components/*.yaml"]
    arch_files = []
    for file in arch_files_wildcard:
      arch_files += glob.glob(file)
    arch_tmp_dir = _simba_dir + "/tmp_arch"
    ds_path = _simba_dir + "/design_space.yaml"
    self.ds = DesignSpace(arch_files, ds_path, workload_dir, tmp_output_dir, arch_tmp_dir)
    super().__init__(n_var=self.ds.dims, n_obj=3, n_ieq_constr=0, xl=self.ds.lb, xu=self.ds.ub)

  def _evaluate(self, x, out, *args, **kwargs):
    out["F"] = self.ds(x)


if __name__ == "__main__":
  networks = ["resnet50", "vgg16", "efficientnetb0", "inceptionv3"] # ["resnet50", "vgg16", "efficientnetb0", "inceptionv3"]
  output_dir = f"{_simba_dir}/outputs/nsga"
  for network in networks:
    print(f" ==== {network} ====")
    workload_dir = f"{_simba_dir}/workloads/{network}"
    tmp_output_dir = f"{_simba_dir}/tmp_outputs/{network}"
    problem = MyProblem(workload_dir, tmp_output_dir)
    algorithm = NSGA2(pop_size=6, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", 10), verbose=True, seed=35)
    pareto_set = [res.X, res.F]
    problem.ds.dump_pareto(pareto_set, output_dir)