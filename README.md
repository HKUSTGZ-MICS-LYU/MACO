## MACO: a HW-Mapping Co-optimization framework for DNN Accelerators
### Introduction
MACO is a HW-Mapping co-optimization framework for DNN accelerators. It uses 
different search algorithms for hardware space exploration and uses the MIP model
for mapping space exploration. We use Timeloop and Accelergy for evaluation.   
The hardware space explorations algorithms include:
* MOBO
* NSGA-II
* Random Search

### Installation
Install Timeloop and Accelergy
```
Timeloop: https://github.com/NVlabs/timeloop
Accelergy: https://github.com/Accelergy-Project/accelergy
```
Install Gurobi. If you are a student, you can use the academic license.
```
Gurobi: https://www.gurobi.com/
```
Install LEMON, a MIP model for the Simba-like chiplet.
```
LEMON: https://github.com/Haimrich/lemon
```
Clone the repository
```
git clone repo_link
```
### Run
In the src folder, there are some scripts, like "run_mobo.py", "run_nsga.py" and "run_random.py". You can run any of them.
### Authors
* Wujie Zhong ([GitHub](https://github.com/zhongwujie))
* Zijun Jiang ([GitHub](https://github.com/Jzjerry))
* Yangdi Lyu ([GitHub](https://github.com/lvyangdi))([Personal Webpage](https://personal.hkust-gz.edu.cn/yangdilyu/index.html))

### Citation
If you use this repository, please cite our paper:
```
@inproceedings{zhong2025maco,
  title={MACO: A Hardware-Mapping Co-Optimization Framework for DNN Accelerators},
  author={Zhong, Wujie and Jiang, Zijun and Lyu, Yangdi},
  booktitle={2025 30th Asia and South Pacific Design Automation Conference (ASP-DAC)},
  year={2025},
  organization={IEEE}
}
```