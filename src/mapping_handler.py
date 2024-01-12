import numpy as np
import re
import json, yaml
import os
import pandas as pd

class LemonHandler:
  # Extract Timeloop Statistics from output files
  def extract_tl_stats(self, file_path: str) -> tuple:
    with open(file_path, "r") as f:
      stats = f.read()
      res = re.findall(r'Cycles:\s(\d+)', stats, re.DOTALL)
      cycles = int(res[0])
      res = re.findall(r'Energy:\s(\d+.\d+)\suJ', stats, re.DOTALL)
      energy = float(res[0])
      res = re.findall(r'Area:\s(\d+.\d+)\smm\^2', stats, re.DOTALL)
      area = float(res[0])
      f.close()
    return cycles, energy, area

  # get the data for a single network
  def get_network_data(self, network_folder: str, loopup_path: str) -> tuple:
    network = network_folder.split('/')[-1]
    with open(loopup_path, 'r') as f:
      lookup_yaml = yaml.safe_load(f)
    layer_workload_lookup = lookup_yaml['layer_workload_lookups'][network]
    workload_ids = sorted(list(set(layer_workload_lookup)))
    
    total_energy = total_latency = 0
    total_area = 0
    for w in layer_workload_lookup:
      stats_file = f'{network_folder}/{str(w).zfill(len(str(len(workload_ids))))}/timeloop-model.stats.txt'
      if not os.path.isfile(stats_file):
        print(f"[ERROR] Missing file: {stats_file}")
      cycles, energy, area = self.extract_tl_stats(stats_file)
      total_energy += energy
      total_latency += cycles
      total_area = max(total_area, area)
    
    edp = total_energy * total_latency

    return (total_energy, total_latency, edp, total_area)