# RL-Scheduling-2022-Summer
Reinforcement Learning approach to solve the Flexible Job Shop Scheduling Problem.

==============================

An optimized OpenAi Gym's environment to simulate the Flexible Job Shop Scheduling Problem.

![til](readme_presentation.gif)

Getting Started
------------

This repository contains a environment for FJSSP as the JSSP file. We need to install the gym package first to use the environment.

```shell
pip install gym
```

Once installed and put JSSP file under the same folder as your main python file, the environment will be available in your OpenAi Gym environment and can be used to train a reinforcement learning agent:

```python
import gym
import JSSP
env = gym.make("JSSP-v0", instance_path = INSTANCE_PATH)
```

### Important: Your instance must follow [section 2.6.1 of this paper](https://ai.vub.ac.be/wp-content/uploads/2019/12/A-Generic-Multi-Agent-Reinforcement-Learning-Approach-for-Scheduling-Problems.pdf). 


How To Use
------------

The observation provided by the environment contains both a boolean array indicating if the action is legal or not and the "real" observation

The `observation` is a tuple of `self.state`, and `self.state` is a dictionary of two entries:  
`job_machine_allocation` : array of integers representing ith job's allocation, where -1 represents an empty allocation and -2 represents a finished job.  
`job_operation_status` : array of integers representing ith job's current operation status.  
For example,  
`observation` = (-1,0,1,0,0,1)  
`self.state` = {  
                `job_machine_allocation`: [-1,0,1]  
                `job_operation_status`: [0,0,1]  
            }  
job 1 (operation 1) -> None  
job 2 (operation 1) -> machine 1  
job 3 (operation 2) -> machine 2  

For each observation, the `action space` is `spaces.Discrete(n)` where n is the number of possible actions.

Project Organization
------------

    ┌── README.md             <- The top-level README for developers using this project.
    │
    ├── JSSP                  <- Contains the environment.
    │
    ├── Instances             <- Contains many example instances of FJSSP.
    │
    ├── GA                    <- Contains supporting files for genetic algorithm.
    │
    └── Solutions
        │
        ├── Solution_GA.ipynb                   <- The genetic algorithm for FJSSP.
        │
        ├── Solution_OR.ipynb                   <- The OR-tool algorithm for FJSSP.
        │
        ├── Solution_prepopulated_RL.ipynb      <- The prepopulated Q-table algorithm.
        │
        └── Solution_instance_division.ipynb    <- The instance division algorithm for FJSSP.
--------

## Credit

The `README.md` file presentation is inspired by [this github repository](https://github.com/prosysscience/JSSEnv), and the genetic algorithm in this repository is from [this blog](https://blog.csdn.net/crazy_girl_me/article/details/118157629).

## Contributors

Name: Hongjian Zhou  
University: University of Oxford  
Email: hongjian.zhou@exeter.ox.ac.uk  

Name: Boyang Gu  
University: Imperial College London  
Email: boyang.gu19@imperial.ac.uk

Name: Chenghao Jin  
Email: steven_jin_gbhg@foxmail.com
