#!/usr/bin/env python
# coding: utf-8

import os

import click
import gym
import numpy as np
import psutil
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging
import tuneconfig

from tfmpc import envs
import tfmpc.solvers.lqr
from tfmpc.solvers import ilqr_run


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

gym.logger.set_level(gym.logger.ERROR)
tf_logging.set_verbosity(tf_logging.ERROR)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("initial-state")
@click.option(
    "--action-size", "-a",
    type=click.IntRange(min=1),
    default=1,
    help="The number of action variables.")
@click.option(
    "--horizon", "-hr",
    type=click.IntRange(min=1),
    default=10,
    help="The number of timesteps.")
@click.option(
    "--debug",
    is_flag=True,
    help="Debug flag.")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbosity flag.")
def lqr(initial_state, action_size, horizon, debug, verbose):
    """Generate and solve a randomly-created LQR problem.

    Args:

        initial_state: list of floats.
    """

    if verbose:
        tf_logging.set_verbosity(tf_logging.INFO)

    if debug:
        tf_logging.set_verbosity(tf_logging.DEBUG)

    initial_state = list(map(float, initial_state.split()))
    x0 = np.array(initial_state, dtype=np.float32)[:,np.newaxis]
    state_size = len(initial_state)

    solver = envs.make_lqr(state_size, action_size)
    trajectory = solver.solve(x0, horizon)

    print(repr(trajectory))
    print()
    print(str(trajectory))


@cli.command()
@click.argument("initial-state")
@click.argument("goal")
@click.option(
    "--beta", "-b",
    type=float,
    default=1.0,
    help="The weight of the action cost.")
@click.option(
    "--horizon", "-hr",
    type=click.IntRange(min=1),
    default=10,
    help="The number of timesteps.")
@click.option(
    "--debug",
    is_flag=True,
    help="Debug flag.")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbosity flag.")
def navlin(initial_state, goal, beta, horizon, debug, verbose):
    """Generate and solve the linear navigation LQR problem.

    Args:

        initial_state: list of floats.

        goal: list of floats.
    """

    if verbose:
        tf_logging.set_verbosity(tf_logging.INFO)

    if debug:
        tf_logging.set_verbosity(tf_logging.DEBUG)

    initial_state = list(map(float, initial_state.split()))
    x0 = np.array(initial_state, dtype=np.float32)[:,np.newaxis]

    goal = list(map(float, goal.split()))
    g = np.array(goal, dtype=np.float32)[:,np.newaxis]

    solver = envs.make_lqr_linear_navigation(g, beta)
    trajectory = solver.solve(x0, horizon)

    print(repr(trajectory))
    print()
    print(str(trajectory))


@cli.command()
@click.argument("env", type=click.Path(exists=True))
@click.option(
    "--horizon", "-hr",
    type=click.IntRange(min=1),
    default=10,
    help="The number of timesteps.",
    show_default=True)
@click.option(
    "--atol",
    type=click.FloatRange(min=0.0),
    default=5e-3,
    help="Absolute tolerance for convergence.",
    show_default=True)
@click.option(
    "--max-iterations", "-miter",
    type=click.IntRange(min=1),
    default=100,
    help="Maximum number of iterations.",
    show_default=True)
@click.option(
    "--logdir",
    type=click.Path(),
    default="/tmp/ilqr/",
    help="Directory used for logging results.",
    show_default=True)
@click.option(
    "--num-samples", "-ns",
    type=click.IntRange(min=1),
    default=1,
    help="Number of runs.",
    show_default=True)
@click.option(
    "--num-workers", "-nw",
    type=click.IntRange(min=1, max=psutil.cpu_count()),
    default=1,
    help=f"Number of worker processes (min=1, max={psutil.cpu_count()}).",
    show_default=True)
@click.option(
    "--verbose", "-v",
    count=True,
    help="Verbosity level flag.")
def ilqr(**kwargs):
    """Run iLQR for a given environment and horizon.

    Args:

        ENV: Path to the environment's config JSON file.
    """
    verbose = kwargs["verbose"]

    if verbose == 1:
        level = tf_logging.INFO
    elif verbose == 2:
        level = tf_logging.DEBUG
    else:
        level = tf_logging.ERROR

    tf_logging.set_verbosity(level)

    def format_fn(param):
        fmt = {
            "env": None,
            "logdir": None,
            "num_samples": None,
            "num_workers": None,
            "verbose": None
        }
        return fmt.get(param, param)

    config_it = tuneconfig.ConfigFactory(kwargs, format_fn)

    runner = tuneconfig.Experiment(config_it, kwargs["logdir"])
    runner.start()

    results = runner.run(ilqr_run, kwargs["num_samples"], kwargs["num_workers"])

    for trial_id, runs in results.items():
        for _, trajectory in runs:
            print(repr(trajectory))
            print(str(trajectory))
