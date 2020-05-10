#!/usr/bin/env python
# coding: utf-8

import click
import gym
import numpy as np
import os
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc import envs
from tfmpc import problems

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
def lqr(initial_state, action_size, horizon):
    """Generate and solve a randomly-created LQR problem.

    Args:

        initial_state: list of floats.
    """
    initial_state = list(map(float, initial_state.split()))
    x0 = np.array(initial_state, dtype=np.float32)[:,np.newaxis]
    state_size = len(initial_state)

    problem = problems.make_lqr(state_size, action_size)
    trajectory = tfmpc.solvers.lqr.solve(problem, x0, horizon)

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
def navlin(initial_state, goal, beta, horizon):
    """Generate and solve the linear navigation LQR problem.

    Args:

        initial_state: list of floats.

        goal: list of floats.
    """
    initial_state = list(map(float, initial_state.split()))
    x0 = np.array(initial_state, dtype=np.float32)[:,np.newaxis]

    goal = list(map(float, goal.split()))
    g = np.array(goal, dtype=np.float32)[:,np.newaxis]

    problem = problems.make_lqr_linear_navigation(g, beta)
    trajectory = tfmpc.solvers.lqr.solve(problem, x0, horizon)

    print(repr(trajectory))
    print()
    print(str(trajectory))


@cli.command()
@click.argument("env")
@click.argument("initial-state")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to the environment's config JSON file.")
@click.option(
    "--horizon", "-hr",
    type=click.IntRange(min=1),
    default=10,
    help="The number of timesteps.",
    show_default=True)
@click.option(
    "--atol",
    type=click.FloatRange(min=0.0),
    default=1e-4,
    help="Absolute tolerance for convergence.",
    show_default=True)
@click.option(
    "--debug",
    is_flag=True,
    help="Debug flag.")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbosity flag.")
def ilqr(env, initial_state, config, horizon, atol, debug, verbose):
    """Run iLQR for the given environment and config.

    Args:

        ENV: environment name.

        INITIAL_STATE: list of floats.
    """

    if verbose:
        tf_logging.set_verbosity(tf_logging.INFO)

    if debug:
        tf_logging.set_verbosity(tf_logging.DEBUG)

    env = envs.make_env(env, config)
    print(env)
    print()

    initial_state = list(map(float, initial_state.split()))
    state_dim = len(initial_state)
    x0 = tf.constant(initial_state, dtype=tf.float32, shape=[state_dim, 1])

    T = tf.constant(horizon)

    solver = tfmpc.solvers.ilqr.iLQR(env, atol=atol)
    trajectory, iterations = solver.solve(x0, T)

    print(repr(trajectory))
    print()
    print(str(trajectory))
