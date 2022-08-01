import gym
import numpy as np
import plotly.figure_factory as ff
import pandas as pd
import random
import datetime

class JSSPEnv(gym.Env):
    def __init__(self, env_data):

        """
        env_data: n+1 rows
        first row has two entries: n and m representing n jobs and m machiens
        then each row represents the information for each job
        """

        # initial values for variables used for instance
        self.data = env_data
        self.jobs, self.machines = self.data[0][0], self.data[0][1]
        self.jobs_history = [[] for _ in range(self.jobs)]
        self.machines_status = self.get_machines_status()
        self.time = 0
        self.operation_times, self.jobs_operations= self.get_operation()
        self.jobs_left_operations = self.jobs_operations
        self.jobs_finished_operations = np.array([0] * self.jobs)
        self.jobs_status = np.array([-1] * self.jobs)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]

        # n x 1 array, each entry represents the action to the job, -1 means do nothing, m means go to machine m
        self.action_space = self.set_action_space(self.jobs, self.machines)

        self.observation_space = gym.spaces.Dict(
            {
                "active": gym.spaces.Box(low=np.zeros(self.jobs), high=np.ones(self.jobs), dtype = int),
                "left_operations": gym.spaces.Box(low=np.ones(self.jobs), high=self.jobs_left_operations, dtype = int),
            }
        )

    def get_obs(self):
        return {
            "active": (self.jobs_status == -1).astype(int),
            "left_operations": self.jobs_left_operations,
        }

    def get_legal_actions(self):
        """
        returns a list of n entries, each represent a job
        for each entry it lists all machines legal to assign to the job
        """
        legal_actions = []

        for job in range(self.jobs):
            if self.jobs_status[job] == -1:
                legal_actions.append([])
            else:
                legal_action = list(np.where(self.operation_times[job][self.jobs_status[job]] != -1)[0])
                legal_actions.append(legal_action)

        return legal_actions

    def set_action_space(self):

        lowbdd = np.zeros(self.jobs+1).fill(-1)
        highbdd = np.zeros(self.jobs+1).fill(self.machines-1)
        action_space = gym.spaces.Box(low=lowbdd, high=highbdd, dtype = int)

        return action_space

    def get_machines_status(self):
        """
        -1 means unoccupied
        other number(eg. i) means occupied by i-th job
        """
        m = self.data[0][1]
        return np.array([-1]*m)

    def get_operation(self):
        """
        opn: operation no
        opt: operation time
        """
        n, m = self.jobs, self.machines
        opn = np.zeros(n,dtype=int)
        maxnum = self.data[1][0]

        for i in range(n):
            now = 1
            opn[i] = self.data[i+1][0]
            if maxnum < self.data[i+1][0]:
                maxnum = self.data[i+1][0]

        opt = np.zeros((n,maxnum,m),dtype=int)

        for i in range(n):
            # job i+1 ,[0] operation number,[1] 
            now = 1

            for j in range(opn[i]): #j is no. of operation
                op_number = self.data[i+1][now] # numbers of machine of operation
                now = now + 1

                for kk in range(m):
                    opt[i][j][kk] = -1
                for kk in range(op_number):
                    mac_n = self.data[i+1][now]-1

                    now = now + 1
                    opt[i][j][mac_n] = self.data[i+1][now]
                    now = now + 1

        return opt, opn

    def is_illegal(self, action):

        """
        for each job,
        first check if the job is busy while action calls it to go to a machine
        then check if the called machine is busy
        then check if the called machine match with the job current operation
        then check if the machine is called by other jobs in this action
        """

        machines_check = np.zeros(self.machines)

        for job in range(self.jobs):
            
            if self.jobs_status[job] != -1 and action[job] != -1:
                return True

            action_machine = action[job][1]
            if self.machines_status[action_machine] != -1:
                return True

            current_op = self.jobs_finished_operations[job]
            if self.operation_times[job][current_op][action_machine] == -1:
                return True

            if machines_check[action_machine] == 1:
                return True
            else:
                machines_check[action_machine] = 1

        return False
                    
    def step(self, action):

        if self.is_illegal(action):
            return self.get_obs(), np.NINF, False, {}

        reward = -1

        # check if any operation is done
        for job in range(self.jobs):
            job_data = self.jobs_history[job]
            if job_data != []:
                operation_data = job_data[-1]
                if (self.jobs_status[job] != -1) and (self.time == operation_data[2]):
                    self.machines_status[self.jobs_history] = -1
                    self.jobs_status[job] = -1
                    self.jobs_finished_operations[job] += 1

        # update with actions
        for job in range(self.jobs):
            if action[job] != -1:
                o, m = self.jobs_finished_operations[job], action[job]
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

        return self.get_obs(), reward, done, {}

    def reset(self):

        self.jobs_history = [[] for _ in range(self.jobs)]
        self.machines_status.fill(-1)
        self.time = 0
        self.jobs_left_operations = self.jobs_operations
        self.jobs_finished_operations = np.array([0] * self.jobs)
        self.jobs_status = np.array([-1] * self.jobs)

        return self.get_obs()

    def render(self, mode="human"):

        df = []

        for job in range(self.jobs):
            for operation in self.jobs_history[job]:
                dict_op = dict()
                dict_op["Task"] = "Job {}".format(job)
                start_time = operation[1]
                finish_time = np.min(self.time, operation[2])
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_time)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_time)
                dict_op["Resource"] = "Machine {}".format(operation[0])
                df.append(dict_op)

        fig = None

        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Resource",
                colors=self.colors,
                show_colorbar=True,
                group_tasks=True,
            )
            fig.update_yaxes(
                autorange="reversed"
            )  # otherwise tasks are listed from the bottom up

        return fig