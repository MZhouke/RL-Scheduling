import unittest
import gym
import numpy as np

INSTANCE1 = "instance1.txt"
OPERATION_MAP_INSTANCE1 = {0: {0: np.array([10, 15, -1]), 1: np.array([-1, 12, 18])},
                           1: {0: np.array([10, -1, 25]), 1: np.array([25, 18, -1]), 2: np.array([-1, 15, 25])}}
OPERATION_MAP_INSTANCE2 = {0: {0: np.array([10, 15, 8]), 1: np.array([-1, 10, 18])},
                           1: {0: np.array([10, -1, 25]), 1: np.array([25, 18, -1]), 2: np.array([-1, 15, 25])}}
JOB_TOTAL_INSTANCE1 = 2
MACHINE_TOTAL_INSTANCE1 = 3
UPPER_ACTION_INSTANCE1 = 2
LOWER_ACTION_INSTANCE1 = -1
UPPER_ALLOCATION_INSTANCE1 = 2
LOWER_ALLOCATION_INSTANCE1 = -2
UPPER_OPERATION_INSTANCE1 = 2
LOWER_OPERATION_INSTANCE1 = 0
OPERATION_LEN_INSTANCE1 = np.array([2, 3])
JOB_DESCRIPTION_INSTANCE2 = np.array([2, 3, 1, 10, 2, 15, 3, 8, 2, 2, 10, 3, 18])
JOB_INDEX_1 = 0
JOB_MACHINE_ALLOCATION_INSTANCE1 = np.array([-1, -1])
JOB_MACHINE_ALLOCATION_INSTANCE2 = np.array([-1, 0])
JOB_MACHINE_ALLOCATION_INSTANCE3 = np.array([1, -1])
JOB_OPERATION_STATUS_INSTANCE1 = np.array([0, 0])
JOB_OPERATION_STATUS_INSTANCE2 = np.array([0, 0])
JOB_OPERATION_STATUS_INSTANCE3 = np.array([0, 1])
JOB_FINISH_TIME_INSTANCE1 = np.array([0, 0])
JOB_FINISH_TIME_INSTANCE2 = np.array([0, 10])
JOB_FINISH_TIME_INSTANCE3 = np.array([24, 10])
LEGAL_ACTIONS_INSTANCE1 = {
    0: np.array([0, 1]),
    1: np.array([0, 2])
}
LEGAL_ACTIONS_INSTANCE2 = {
    0: np.array([1]),
    1: np.array([])
}
ACTION_1 = np.array([2, -1])
ACTION_2 = np.array([-1, 2])
ACTION_3 = np.array([0, -1])
ACTION_4 = np.array([0, 0])
ACTION_5 = np.array([1, -1])
ACTION_J2_M1 = np.array([-1, 0])
ACTION_J1_M2 = np.array([1, -1])
ACTION_WAIT = np.array([-1, -1])


def generate_env_var(instance_path):
    env_name = "JSSP-v0"
    env = gym.make(env_name, instance_path=instance_path)
    return env


class TestStringMethods(unittest.TestCase):

    def validate_operation_map(self, job_total, operation_len, machine_total,
                               operation_map_desired, operation_map_validate):
        for job_index in range(job_total):
            for operation_index in range(operation_len[job_index]):
                for machine_index in range(machine_total):
                    self.assertEqual(operation_map_desired[job_index][operation_index][machine_index],
                                     operation_map_validate[job_index][operation_index][machine_index])

    def test_initialize(self):
        env = generate_env_var(INSTANCE1)
        env.initialize(INSTANCE1)
        self.assertEqual(env.job_total, JOB_TOTAL_INSTANCE1)
        self.assertEqual(env.machine_total, MACHINE_TOTAL_INSTANCE1)
        self.validate_operation_map(JOB_TOTAL_INSTANCE1,
                                    OPERATION_LEN_INSTANCE1,
                                    MACHINE_TOTAL_INSTANCE1,
                                    OPERATION_MAP_INSTANCE1,
                                    env.job_operation_map)

    def test_populate_job_description_map(self):
        env = generate_env_var(INSTANCE1)
        env.populate_job_description_map(JOB_DESCRIPTION_INSTANCE2, JOB_INDEX_1)
        self.validate_operation_map(JOB_TOTAL_INSTANCE1,
                                    OPERATION_LEN_INSTANCE1,
                                    MACHINE_TOTAL_INSTANCE1,
                                    OPERATION_MAP_INSTANCE2,
                                    env.job_operation_map)

    def test_initialize_action_space(self):
        env = generate_env_var(INSTANCE1)
        for i in range(20):
            sample_action = env.action_space.sample()
            self.assertTrue(np.all(sample_action <= UPPER_ACTION_INSTANCE1))
            self.assertTrue(np.all(sample_action >= LOWER_ACTION_INSTANCE1))

    def test_initialize_obs_space(self):
        env = generate_env_var(INSTANCE1)
        for i in range(20):
            sample_observation = env.observation_space.sample()
            self.assertTrue(np.all(sample_observation[env.job_machine_allocation]
                                   <= UPPER_ALLOCATION_INSTANCE1))
            self.assertTrue(np.all(sample_observation[env.job_machine_allocation]
                                   >= LOWER_ALLOCATION_INSTANCE1))
            self.assertTrue(np.all(sample_observation[env.job_operation_status]
                                   <= UPPER_OPERATION_INSTANCE1))
            self.assertTrue(np.all(sample_observation[env.job_operation_status]
                                   >= LOWER_OPERATION_INSTANCE1))

    def test_get_obs(self):
        env = generate_env_var(INSTANCE1)
        observation = env.get_obs()
        self.assertTrue(np.array_equal(observation[env.job_machine_allocation],
                                       JOB_MACHINE_ALLOCATION_INSTANCE1))
        self.assertTrue(np.array_equal(observation[env.job_operation_status],
                                       JOB_OPERATION_STATUS_INSTANCE1))
        self.assertTrue(np.array_equal(env.job_finish_time,
                                       JOB_FINISH_TIME_INSTANCE1))

    def test_get_legal_actions(self):
        env = generate_env_var(INSTANCE1)
        legal_actions = env.get_legal_actions()
        for i in range(env.job_total):
            self.assertTrue(np.array_equal(legal_actions[i],
                                           LEGAL_ACTIONS_INSTANCE1[i]))
        # send job 2 -> machine 1
        env.update_state(ACTION_J2_M1)
        legal_actions = env.get_legal_actions()
        for i in range(env.job_total):
            self.assertTrue(np.array_equal(legal_actions[i],
                                           LEGAL_ACTIONS_INSTANCE2[i]))

    def test_is_legal(self):
        env = generate_env_var(INSTANCE1)
        # send job 2 -> machine 1
        env.update_state(ACTION_J2_M1)
        self.assertTrue(not (env.is_legal(ACTION_1)))
        self.assertTrue(not (env.is_legal(ACTION_2)))
        self.assertTrue(not (env.is_legal(ACTION_3)))
        self.assertTrue(not (env.is_legal(ACTION_4)))
        self.assertTrue(env.is_legal(ACTION_5))
        self.assertTrue(env.is_legal(ACTION_WAIT))

    def test_update_state(self):
        env = generate_env_var(INSTANCE1)
        # send job 2 -> machine 1
        env.update_state(ACTION_J2_M1)
        self.assertTrue(np.array_equal(env.state[env.job_machine_allocation],
                                       JOB_MACHINE_ALLOCATION_INSTANCE2))
        self.assertTrue(np.array_equal(env.state[env.job_operation_status],
                                       JOB_OPERATION_STATUS_INSTANCE2))
        self.assertTrue(np.array_equal(env.job_finish_time,
                                       JOB_FINISH_TIME_INSTANCE2))
        for i in range(8):
            env.update_time()

        self.assertTrue(env.is_legal(ACTION_5))
        self.assertTrue(not (env.is_legal(ACTION_2)))
        self.assertTrue(not (env.is_legal(ACTION_3)))

        # send job 1 -> machine 3
        env.update_state(ACTION_J1_M2)
        self.assertTrue(np.array_equal(env.state[env.job_machine_allocation],
                                       JOB_MACHINE_ALLOCATION_INSTANCE3))
        self.assertTrue(np.array_equal(env.state[env.job_operation_status],
                                       JOB_OPERATION_STATUS_INSTANCE3))
        self.assertTrue(np.array_equal(env.job_finish_time,
                                       JOB_FINISH_TIME_INSTANCE3))

        self.assertTrue(not (env.is_legal(ACTION_2)))
        self.assertTrue(not (env.is_legal(ACTION_3)))
        self.assertTrue(not (env.is_legal(ACTION_5)))
        # send job wait
        env.update_state(ACTION_WAIT)
        self.assertTrue(np.array_equal(env.state[env.job_machine_allocation],
                                       JOB_MACHINE_ALLOCATION_INSTANCE3))
        self.assertTrue(np.array_equal(env.state[env.job_operation_status],
                                       JOB_OPERATION_STATUS_INSTANCE3))
        self.assertTrue(np.array_equal(env.job_finish_time,
                                       JOB_FINISH_TIME_INSTANCE3))
        self.assertEqual(env.time, 11)

    def test_step(self):
        env = generate_env_var(INSTANCE1)
        initial_observation = env.reset()
        # illegal action
        observation_1, reward_1, done_1, info = env.step(ACTION_1)
        self.assertTrue(np.array_equal(observation_1[env.job_machine_allocation],
                                       JOB_MACHINE_ALLOCATION_INSTANCE1))
        self.assertTrue(np.array_equal(observation_1[env.job_operation_status],
                                       JOB_OPERATION_STATUS_INSTANCE1))
        self.assertTrue(np.array_equal(env.job_finish_time,
                                       JOB_FINISH_TIME_INSTANCE1))
        self.assertEqual(reward_1, env.illegal_reward)
        self.assertEqual(env.time, 0)
        self.assertFalse(done_1)
        # send job 2 -> machine 1
        observation_2, reward_2, done_2, info = env.step(ACTION_J2_M1)
        self.assertTrue(np.array_equal(observation_2[env.job_machine_allocation],
                                       JOB_MACHINE_ALLOCATION_INSTANCE2))
        self.assertTrue(np.array_equal(observation_2[env.job_operation_status],
                                       JOB_OPERATION_STATUS_INSTANCE2))
        self.assertTrue(np.array_equal(env.job_finish_time,
                                       JOB_FINISH_TIME_INSTANCE2))
        self.assertEqual(reward_2, -1)
        self.assertEqual(env.time, 1)
        self.assertFalse(done_2)

    def test_reset(self):
        env = generate_env_var(INSTANCE1)
        initial_observation = env.reset()
        env.step(ACTION_J2_M1)
        observation_after_reset = env.reset()
        self.assertTrue(np.array_equal(observation_after_reset[env.job_machine_allocation],
                                       JOB_MACHINE_ALLOCATION_INSTANCE1))
        self.assertTrue(np.array_equal(observation_after_reset[env.job_operation_status],
                                       JOB_OPERATION_STATUS_INSTANCE1))
        self.assertTrue(np.array_equal(env.job_finish_time,
                                       JOB_FINISH_TIME_INSTANCE1))


if __name__ == '__main__':
    unittest.main()
