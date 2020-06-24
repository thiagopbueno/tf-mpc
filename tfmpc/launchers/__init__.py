import json
import os

import tensorflow as tf

from tfmpc import agents
from tfmpc import envs
from tfmpc import runners
from tfmpc.solvers import ilqr
from tfmpc.loggers import get_logger


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


def online_ilqr_run(config):
    import os

    import psutil

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))

    env_config = config.pop("env")
    with open(env_config, "r") as file:
        env_config = json.load(file)

    env = envs.make_env(env_config)

    x0 = tf.constant(env_config["initial_state"], dtype=tf.float32)
    T = tf.constant(config.pop("horizon"), dtype=tf.int32)

    logger = get_logger(config.get("logger", "py_logging"), config["logdir"])
    config["logger"] = logger

    solver = ilqr.iLQR(env, **config)
    controller = agents.MPC(solver, T)

    runner = runners.Runner(env, controller,
                            on_step=logger.log_transition,
                            on_episode_end=logger.summary)

    with runner(x0, T) as r:
        trajectory = r.run()
        output = os.path.join(config["logdir"], "data.csv")
        trajectory.save(output)

    return env, trajectory

