import json
import os

import tensorflow as tf

from tfmpc import envs
from tfmpc.solvers import ilqr


def ilqr_run(config):
    env_config = config.pop("env")
    with open(env_config, "r") as file:
        env_config = json.load(file)

    env = envs.make_env(env_config)

    x0 = tf.constant(env_config["initial_state"], dtype=tf.float32)
    T = tf.constant(config.pop("horizon"), dtype=tf.int32)

    solver = ilqr.iLQR(env, **config)
    trajectory, iterations = solver.solve(x0, T)

    output = os.path.join(config["logdir"], "data.csv")
    trajectory.save(output)

    return env, trajectory
