import gym
import numpy as np
import plotly.figure_factory as ff
import pandas as pd
import random
import datetime


class JSSPEnv(gym.Env):
    def __init__(self, instance_path):

        """
        env_data: n+1 rows
        first row has two entries: n and m representing n jobs and m machiens
        then each row represents the information for each job
        """

        # initial values for variables used for instance
        # job_total is total # of jobs
        # machine_total is total # of machines
        # job operation map stores the description of the JSP
        self.job_total, self.machine_total, self.job_operation_map = 0, 0, {}
        self.initialize(instance_path)

        self.jobs_history = [[] for _ in range(self.job_total)]
        self.machines_status = self.get_machines_status()
        self.time = 0
        self.operation_times, self.jobs_operations = [],[]
        self.jobs_left_operations = self.jobs_operations
        self.jobs_finished_operations = np.array([0] * self.job_total)
        self.jobs_status = np.array([-1] * self.job_total)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machine_total)
        ]

        # n x 1 array, each entry represents the action to the job, -1 means do nothing, m means go to machine m
        self.action_space = self.set_action_space()

        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "active": gym.spaces.Box(low=np.zeros(self.job_total), high=np.ones(self.job_total), dtype=int),
        #         "left_operations": gym.spaces.Box(low=np.ones(self.job_total), high=self.jobs_left_operations, dtype=int),
        #     }
        # )

    def initialize(self, instance_path):
        """
            populate job_operation_map using the instance file input
            :param instance_path: a string representing file path
        """
        # input instance description for environment initialization
        instance_path = instance_path
        file_handle = open(instance_path, 'r')
        lines_list = file_handle.readlines()
        # first line consists of # of jobs in total, # of machines in total
        self.job_total, self.machine_total = [int(x) for x in lines_list[0].split()]
        # env_data = [[int(val) for val in line.split()] for line in lines_list[1:]]
        # read through each job description
        for job_index in range(len(lines_list) - 1):
            job_description = [int(val) for val in lines_list[job_index + 1].split()]
            self.job_operation_map[job_index] = {}
            # initialize job_operation_map
            # ex. job_operation_map[1][2][3] = time it takes for 3rd machine to execute 2nd operation of 1st job
            # time = -1 iff the machine is not capable of executing the operation based on instance description
            for operation_index in range(job_description[0]):
                self.job_operation_map[job_index][operation_index] = np.negative(np.ones(self.machine_total))
            # populate job_description_map
            self.populate_job_description_map(job_description, job_index)

    def populate_job_description_map(self, job_description, job_index):
        """
            populate the corresponding fields in job_description_map
            :param job_description: a string representing job description
            :param job_index: an integer representing current job index being populated
        """
        # read job description from left to right
        description_pivot = 1
        operation_index = 0
        # operation_index
        while description_pivot < len(job_description):
            # operation_description length = 2 * # of machines capable of executing current operation at the pivot
            operation_description_end = 2 * job_description[description_pivot] + description_pivot
            # read the current description of operation
            description_pivot += 1
            while description_pivot <= operation_description_end:
                # Following the format of operation description:
                # machine index , time it takes for the current operation
                machine_index = job_description[description_pivot]
                operation_duration = job_description[description_pivot + 1]
                self.job_operation_map[job_index][operation_index][machine_index - 1] = operation_duration
                description_pivot += 2
            operation_index += 1

    def get_obs(self):
        """
        :return: observation from the current action
        """
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

        for job in range(self.job_total):
            if self.jobs_status[job] == -1:
                legal_actions.append([])
            else:
                legal_action = list(np.where(self.operation_times[job][self.jobs_status[job]] != -1)[0])
                legal_actions.append(legal_action)

        return legal_actions

    def set_action_space(self):

        lowbdd = np.full(self.job_total, -1)
        highbdd = np.full(self.job_total, self.machine_total - 1)
        action_space = gym.spaces.Box(low=lowbdd, high=highbdd, dtype=int)

        return action_space

    def get_machines_status(self):
        """
        -1 means unoccupied
        other number(eg. i) means occupied by i-th job
        """
        return np.array([-1] * self.machine_total)

    def get_operation(self):
        """
        opn: operation no
        opt: operation time
        """
        # n, m = self.job_total, self.machine_total
        # opn = np.zeros(n, dtype=int)
        # maxnum = self.data[1][0]
        #
        # for i in range(n):
        #     now = 1
        #     opn[i] = self.data[i + 1][0]
        #     if maxnum < self.data[i + 1][0]:
        #         maxnum = self.data[i + 1][0]
        #
        # opt = np.zeros((n, maxnum, m), dtype=int)
        #
        # for i in range(n):
        #     # job i+1 ,[0] operation number,[1]
        #     now = 1
        #
        #     for j in range(opn[i]):  # j is no. of operation
        #         op_number = self.data[i + 1][now]  # numbers of machine of operation
        #         now = now + 1
        #
        #         for kk in range(m):
        #             opt[i][j][kk] = -1
        #         for kk in range(op_number):
        #             mac_n = self.data[i + 1][now] - 1
        #
        #             now = now + 1
        #             opt[i][j][mac_n] = self.data[i + 1][now]
        #             now = now + 1

        return

    def is_illegal(self, action):

        """
        for each job,
        first check if the job is busy while action calls it to go to a machine
        then check if the called machine is busy
        then check if the called machine match with the job current operation
        then check if the machine is called by other jobs in this action
        """

        machines_check = np.zeros(self.machine_total)

        for job in range(self.job_total):

            if self.jobs_status[job] != -1 and action[job] != -1:
                return True

            action_machine = action[job]
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
        for job in range(self.job_total):
            job_data = self.jobs_history[job]
            if job_data:
                operation_data = job_data[-1]
                if (self.jobs_status[job] != -1) and (self.time == operation_data[2]):
                    self.machines_status[self.jobs_history] = -1
                    self.jobs_status[job] = -1
                    self.jobs_finished_operations[job] += 1

        # update with actions
        for job in range(self.job_total):
            if action[job] != -1:
                o, m = self.jobs_finished_operations[job], action[job]
                job_history = self.jobs_history[job]
                job_history.append([m, self.time, self.time + self.operation_times[job][o][m]])
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

        self.jobs_history = [[] for _ in range(self.job_total)]
        self.machines_status.fill(-1)
        self.time = 0
        self.jobs_left_operations = self.jobs_operations
        self.jobs_finished_operations = np.array([0] * self.job_total)
        self.jobs_status = np.array([-1] * self.job_total)

        return self.get_obs()

    def render(self, mode="human"):

        df = []

        for job in range(self.job_total):
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
