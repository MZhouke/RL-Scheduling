from gym.envs.registration import register

register(
        id='JSSP-v0',
        entry_point='gym_JSSP.envs:JSSPEnv',
        )