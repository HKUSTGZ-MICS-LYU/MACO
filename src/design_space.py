import yaml
import numpy as np
import os, shutil
import subprocess
from typing import Dict, List, Tuple
# import from the project
from accelergy_handler import AccelergyHandler
from accelergy.utils import accelergy_dumper
from mapping_handler import LemonHandler
import utils
import time

class DesignSpace:
  def __init__(self, arch_files: List[str], ds_path: str, workload_dir: str, tmp_output_dir: str,
      arch_tmp_dir: str):
    with open(ds_path, 'r') as f:
      self.design_space: dict = yaml.safe_load(f)
    self.dim2names = {} # dim_id : [GlobalBuffer, depth]
    self.dim2slice = {} # slice is the min, max, slice_num, default dictionary
    self.syn_depth = {} # src_name : dest_name, src follows the value of dest
    self.syn_instances = {}
    self.keep_depth = {} # src_name : value, src keeps the value
    self.keep_instances = {}
    self.dims = None
    self.accelergy_handler = AccelergyHandler(arch_files)
    self.workload_dir = workload_dir
    self.tmp_output_dir = tmp_output_dir
    self.arch_tmp_dir = arch_tmp_dir
    self.init()
    
    self.lb = np.zeros(self.dims)
    self.ub = self.get_ub()
    self.default_cand = self.get_default_cand()
    self.default_area = self.calc_area(self.default_cand)
  
  # initialize dim2names, syn_depth, syn_instances and dims
  def init(self):
    total_length = 0
    block_list = self.design_space['design_space']
    
    for block in block_list:
      assert len(block) == 1
      block_key = next(iter(block.keys()))
      block_values = next(iter(block.values())) # list
      for ds_entry in block_values: # ds_entry: "depth" or "instances" dict
        assert len(ds_entry) == 1
        names = [block_key]
        ds_key = next(iter(ds_entry.keys()))
        ds_entry_values = next(iter(ds_entry.values())) # min_max dict
        names.append(ds_key)
        if "syncronize" in ds_entry_values:
          dest_name = ds_entry_values["syncronize"]
          if ds_key == "depth":
            self.syn_depth[block_key] = dest_name
          elif ds_key == "instances":
            self.syn_instances[block_key] = dest_name
        elif "keep" in ds_entry_values:
          keep_val = ds_entry_values["keep"]
          if ds_key == "depth":
            self.keep_depth[block_key] = keep_val
          elif ds_key == "instances":
            self.keep_instances[block_key] = keep_val
        else:
          self.dim2names[total_length] = names
          self.dim2slice[total_length] = ds_entry_values
          total_length += 1
          
    self.dims = total_length
  
  def gen_depth_updates(self, candidate: List[int]) -> Dict[str, int]:
    updates = {}
    for dim_id in range(len(candidate)):
      names = self.dim2names[dim_id]
      if "depth" in names:
        level_name = names[0]
        space_list = self.get_space_list(dim_id)
        level_depth = space_list[candidate[dim_id]]
        updates[level_name] = level_depth
    
    for key, value in self.syn_depth.items():
      updates[key] = updates[value]
    for key, value in self.keep_depth.items():
      updates[key] = value
    
    return updates

  def gen_instances_updates(self, candidate: List[int]) -> Dict[str, int]:
    updates = {}
    for key, value in self.keep_instances.items():
      updates[key] = value
    for dim_id in range(len(candidate)):
      names = self.dim2names[dim_id]
      if "instances" in names:
        name = names[0]
        space_list = self.get_space_list(dim_id)
        instances_num = space_list[candidate[dim_id]]
        updates[name] = instances_num
    
    for key, value in self.syn_instances.items():
      updates[key] = updates[value]
    return updates
      
  def get_space_list(self, dim_id: int) -> List[int]:
    slice_dict = self.dim2slice[dim_id] # dict
    step = (slice_dict["max"] - slice_dict["min"]) / slice_dict["slice_num"]
    ret_list = [int(slice_dict["min"] + step * (i + 1)) for i in range(slice_dict["slice_num"])]
    return ret_list
  
  def get_ub(self) -> np.ndarray:
    ub = np.zeros(self.dims)
    for dim_id in range(self.dims):
      ub[dim_id] = len(self.get_space_list(dim_id))
    return ub
  
  def calc_area(self, cand: List[int]) -> float:
    # get updates
    depth_updates = self.gen_depth_updates(cand)
    inst_updates = self.gen_instances_updates(cand)
    # call accelergy handler
    self.accelergy_handler.update_accelergy(depth_updates, inst_updates)
    ert, art = self.accelergy_handler.get_ert_art()
    # calculate area
    area = self.gen_area_from_art(art, inst_updates)
    return area
  
  def gen_area_from_art(self, art: dict, insts: Dict[str, int]) -> float:
    area = 0
    tables = art["ART"]["tables"]
    for table in tables:
      name = table["name"]
      name = name.split(".")[-1]
      if name in insts:
        area += table["area"] * insts[name]
    return area
  
  def get_default_cand(self) -> List[int]:
    default_cand = [0] * self.dims
    for dim_id in range(self.dims):
      slice_dict = self.dim2slice[dim_id]
      default_val = slice_dict["default"]
      space_list = self.get_space_list(dim_id)
      defalut_id = space_list.index(default_val)
      default_cand[dim_id] = defalut_id
    return default_cand
  
  def __call__(self, x: np.ndarray) -> List[float]:
    energy: float = 2**63
    cycles: float = 2**63
    area: float = 2**63
    cand = list(x.astype(int))
    cand_area = self.calc_area(cand) # update accelergy in the mean time
    if(cand_area > self.default_area):
      return [energy, cycles, area]
    
    # generate tmp arch file
    new_arch = self.accelergy_handler.get_arch()
    temp_arch_path = self.arch_tmp_dir + "/simba_v3.yaml"
    with open(temp_arch_path, "w") as f:
      yaml.dump(new_arch, f, default_flow_style=False, Dumper=accelergy_dumper)
      
    # run lemon
    arch_mapspace_files = f"{self.arch_tmp_dir}/*.yaml" + f" {self.arch_tmp_dir}/components/*.yaml"
    lookup_path = os.path.dirname(self.workload_dir) + "/lookups.yaml"
    network = self.workload_dir.split('/')[-1]
    if os.path.exists(self.tmp_output_dir):
      shutil.rmtree(self.tmp_output_dir)
    os.mkdir(self.tmp_output_dir)
    with open(lookup_path, "r") as f:
      lookup_yaml = yaml.safe_load(f)
    lookup_list = lookup_yaml['layer_workload_lookups'][network]
    lookup_str = " ".join(map(str, lookup_list))
    cmd = f"lemon {arch_mapspace_files} {self.workload_dir} -lu {lookup_str} -o {self.tmp_output_dir}"
    try:
      begin = time.time()
      result = subprocess.run(cmd.split(), capture_output=True, timeout=1500)
      end = time.time()
      print(f"lemon takes {end - begin} seconds", flush=True)
    except subprocess.TimeoutExpired:
      print("timeout after 1500 seconds", flush=True)
    success = utils.check_suffix_file(self.tmp_output_dir, ".stats.txt")
    if not success: # lemon fails or timeout
      return [energy, cycles, area]
      # print(result.stdout.decode("utf-8"))
      # print(result.stderr.decode("utf-8"))
      # exit(1)
    # get energy, cycles and area
    lemon_handler = LemonHandler()
    energy, cycles, edp, area = lemon_handler.get_network_data(self.tmp_output_dir, lookup_path)
    return [round(energy, 2), round(cycles, 2), area]

  def gen_micro_arch(self, cand: List[int]) -> dict:
    # get updates
    depth_updates = self.gen_depth_updates(cand)
    depth_updates_dict = dict(depth_updates)
    inst_updates = self.gen_instances_updates(cand)
    micro_arch = {}
    for name, inst in inst_updates.items():
      if name in depth_updates_dict:
        depth = depth_updates_dict[name]
        micro_arch[name] = {"depth": depth, "instances": inst}
      else:
        micro_arch[name] = {"instances": inst}
    return micro_arch
  
  def dump_pareto(self, samples: np.ndarray, output_dir: str):
    X = samples[0]
    Y = samples[1]
    pareto_num = len(X)
    print(f"find {pareto_num} pareto points")
    network = self.workload_dir.split('/')[-1]
    output_folder = f"{output_dir}/{network}"
    pareto_dir = os.path.join(output_folder, "pareto")
    pareto_csv_path = os.path.join(output_folder, "pareto.csv")
    if os.path.exists(pareto_dir):
      shutil.rmtree(pareto_dir)
    os.makedirs(pareto_dir)
      
    with open(pareto_csv_path, "w") as f:
      f.write("id, energy, cycles, area\n")
      for i in range(pareto_num):
        f.write(f"{i}, {Y[i][0]}, {Y[i][1]}, {Y[i][2]}\n")
        
    for i in range(pareto_num):
      update_path = os.path.join(pareto_dir, f"update_{i}.yaml")
      x = X[i]
      cand = list(x.astype(int))
      micro_arch = self.gen_micro_arch(cand)
      with open(update_path, "w") as f:
        yaml.dump(micro_arch, f, default_flow_style=False, sort_keys=False)