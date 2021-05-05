from gym.envs.registration import register

register(
    id='ManipEnv-v0',
    entry_point='manip_envs.tabletop:Tabletop',
)
