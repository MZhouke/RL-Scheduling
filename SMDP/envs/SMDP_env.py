import gym
import numpy as np
from gym import spaces


class SMDPEnv(gym.Env):
    def __init__(self, env_data):

        """
        env_data: n+1 rows
        first row has two entries: n and m representing n jobs and m machiens
        then each row represents the information for each job
        """

        # initial values for variables used for instance
        self.jobs, self.machines = env_data[0][0], env_data[0][1]
        self.jobs_history = [[] for _ in range(self.jobs)]
        self.machines_status = self.get_machines_status(env_data)
        self.time = 0
        self.operation_times, self.jobs_operations= self.get_operation(env_data)
        self.jobs_left_operations = self.jobs_operations
        self.jobs_finished_operations = [0] * self.jobs
        self.jobs_status = [-1] * self.jobs

        # no. of left operations for each job
        self.observation_space = gym.spaces.Box(low=np.ones(self.jobs), high=self.jobs_left_operations, dtype = int)

        # each row representing one job. The first entry represents whether the job is activated or not, and the second entry represents the machine it assisngs
        self.action_space = self.get_action_space(self.jobs, self.machines)

    def get_action_space(self, jobs, machines):

        lowbdd = np.zeros((jobs, 2))
        highbdd = np.zeros((jobs,2))
        highbdd[:,0] = np.ones(jobs)
        highbdd[:,1] = np.array([machines]*jobs)
        action_space = gym.spaces.Box(low=lowbdd, high=highbdd, dtype = int)

        return action_space

    def get_machines_status(self, data):
        """
        -1 means unoccupied
        other number(eg. i) means occupied by i-th job
        """
        m = data[0][1]
        return np.array([-1]*m)

    def get_operation(self, data):
        """
        opn: operation no
        opt: operation time
        """
        n, m = self.jobs, self.machines
        opn = np.zeros(n,dtype=int)
        maxnum = data[1][0]

        for i in range(n):
            now = 1
            opn[i] = data[i+1][0]
            if maxnum < data[i+1][0]:
                maxnum = data[i+1][0]

        opt = np.zeros((n,maxnum,m),dtype=int)

        for i in range(n):
            # job i+1 ,[0] operation number,[1] 
            now = 1

            for j in range(opn[i]): #j is no. of operation
                op_number = data[i+1][now] # numbers of machine of operation
                now = now + 1

                for kk in range(m):
                    opt[i][j][kk] = -1
                for kk in range(op_number):
                    mac_n = data[i+1][now]-1

                    now = now + 1
                    opt[i][j][mac_n] = data[i+1][now]
                    now = now + 1

        return opt, opn

    def step(self, action):

        reward = -1

        # check if any operation is done
        for job in range(self.jobs):
            job_data = self.jobs_history[job]
            operation_data = job_data[-1]
            if (self.jobs_status[job] != -1) and (self.time == operation_data[2]):
                self.machines_status[self.jobs_history] = -1
                self.jobs_status[job] = -1
                self.jobs_finished_operations[job] += 1

        # update with actions
        for job in range(len(action)):
            if action[job][0] == 1:
                o, m = self.jobs_left_operations, action[job][1]
                if self.operation_times[job][o][m]!=-1 and self.machines_status[m]==-1:
                    job_history = self.jobs_history[job]
                    job_history.append([m, self.time, self.time+self.operation_times[job][o][m]])
                    self.jobs_history[job] = job_history
                    self.machines_status[m] = job
                    self.jobs_status[job] = o
                    self.jobs_left_operations[job] -= 1

        # update time
        self.time += 1

        done = np.all(self.jobs_finished_operations == self.jobs_operations)

        if done:
            reward = 0
            self.time -= 1

        return self.jobs_left_operations, reward, done, []

    def reset(self):
        self.jobs_history = [[] for _ in range(self.jobs)]
        self.machines_status.fill(-1)
        self.time = 0
        self.jobs_left_operations = self.jobs_operations
        self.jobs_finished_operations = [0] * self.jobs
        self.jobs_status = [-1] * self.jobs

    def render(self):
        pass