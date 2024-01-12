import os
import time
import glob
from algorithms.morand.random_method import RandomMethod
from design_space import DesignSpace

_simba_dir = "../examples/simba"

def run(workload_dir: str, tmp_output_dir: str, output_dir: str):
  arch_files_wildcard = [_simba_dir + "/arch/*.yaml", _simba_dir + "/arch/components/*.yaml"]
  arch_files = []
  for file in arch_files_wildcard:
    arch_files += glob.glob(file)
  arch_tmp_dir = _simba_dir + "/tmp_arch"
  ds_path = _simba_dir + "/design_space.yaml"
  ds = DesignSpace(arch_files, ds_path, workload_dir, tmp_output_dir, arch_tmp_dir)
  rm = RandomMethod(nobjs=3, lb=ds.lb, ub=ds.ub, dims=ds.dims, func=ds, neval=60, seed=35)
  begin = time.time()
  rm.start()
  end = time.time()
  print(f"Random takes {end - begin} seconds")
  ds.dump_pareto(rm.curt_pareto_samples, output_dir)


if __name__ == "__main__":
  networks = ["resnet50", "vgg16", "efficientnetb0", "inceptionv3"] # ["resnet50", "vgg16", "efficientnetb0", "inceptionv3"]
  output_dir = f"{_simba_dir}/outputs/random"
  for network in networks:
    print(f" ==== {network} ====")
    workload_dir = f"{_simba_dir}/workloads/{network}"
    tmp_output_dir = f"{_simba_dir}/tmp_outputs/{network}"
    run(workload_dir, tmp_output_dir, output_dir)