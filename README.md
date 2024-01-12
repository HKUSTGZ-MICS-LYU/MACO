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