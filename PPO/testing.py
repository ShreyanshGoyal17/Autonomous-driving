from metadrive import MetaDriveEnv
import random

config = dict(
    use_render=False,
    manual_control=False,
    traffic_density=0.0,
    random_agent_model=False,
    random_lane_width=False,
    random_lane_num=False,
    use_lateral=True,
    # map="SCrROXT",
    # map="OCXT",
    start_seed=random.randint(0, 1000),
    map=7,  # seven block
    environment_num=100,
)


env = MetaDriveEnv(config)


print(env.action_space)