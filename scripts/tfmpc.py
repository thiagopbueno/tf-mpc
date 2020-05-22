#!/usr/bin/env python
# coding: utf-8

import json
import os

import click
import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc import envs

import tfmpc.solvers.lqr
import tfmpc.solvers.ilqr


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
@click.argument("config", type=click.Path(exists=True))
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
    "--verbose", "-v",
    count=True,
    help="Verbosity level flag.")
def ilqr(config, horizon, atol, max_iterations, verbose):
    """Run iLQR for a given environment and horizon.

    Args:

        CONFIG: Path to the environment's config JSON file.
    """

    if verbose == 1:
        level = tf_logging.INFO
    elif verbose == 2:
        level = tf_logging.DEBUG
    else:
        level = tf_logging.ERROR

    tf_logging.set_verbosity(level)

    with open(config, "r") as file:
        config = json.load(file)

    env = envs.make_env(config)
    print(env)
    print()

    x0 = tf.constant(config["initial_state"], dtype=tf.float32)
    T = tf.constant(horizon)

    solver = tfmpc.solvers.ilqr.iLQR(env, atol=atol)
    trajectory, iterations = solver.solve(x0, T)

    print(repr(trajectory))
    print()
    print(str(trajectory))
