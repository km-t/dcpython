from gym.envs.registration import register

register(
    id='myenv_keras-v0',
    entry_point='myenv.env:MyEnv'
)
