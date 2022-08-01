import unittest
import gym
import numpy as np

INSTANCE1 = "instance1.txt"
OPERATION_MAP_INSTANCE1 = {0: {0: np.array([10, 15, -1]), 1: np.array([-1, 12, 18])},
                           1: {0: np.array([10, -1, 25]), 1: np.array([25, 18, -1]), 2: np.array([-1, 15, 25])}}
JOB_TOTAL_INSTANCE1 = 2
MACHINE_TOTAL_INSTANCE1 = 3
OPERATION_LEN_INSTANCE1 = [2, 3]


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
                                     operation_map_validate[job_index][operation_index][machine_index]
                                     )

    def test_initialize(self):
        env = generate_env_var(INSTANCE1)
        self.assertEqual(env.job_total, JOB_TOTAL_INSTANCE1)
        self.assertEqual(env.machine_total, MACHINE_TOTAL_INSTANCE1)
        self.validate_operation_map(JOB_TOTAL_INSTANCE1,
                                    OPERATION_LEN_INSTANCE1,
                                    MACHINE_TOTAL_INSTANCE1,
                                    OPERATION_MAP_INSTANCE1,
                                    env.job_operation_map)

    def test_populate_job_description_map(self):
        assert False

    def test_get_obs(self):
        assert False

    def test_get_legal_actions(self):
        assert False

    def test_set_action_space(self):
        assert False

    def test_get_machines_status(self):
        assert False

    def test_get_operation(self):
        assert False

    def test_is_illegal(self):
        assert False

    def test_step(self):
        assert False

    def test_reset(self, env):
        assert False


if __name__ == '__main__':
    unittest.main()
