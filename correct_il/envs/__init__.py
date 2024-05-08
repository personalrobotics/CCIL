import gym
from gym.envs.registration import register as gymregister

def register(id, entry_point, *args, **kwargs):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        del env_specs[id]
    gymregister(
        id=id,
        entry_point=entry_point,
        *args,
        **kwargs
    )

register(id="PendulumSwingupCont-v0", entry_point="correct_il.envs.pendulum_env:PendulumEnvCont", max_episode_steps=500)
register(id="PendulumSwingupDisc-v0", entry_point="correct_il.envs.pendulum_env:PendulumEnvDiscont", max_episode_steps=500)
