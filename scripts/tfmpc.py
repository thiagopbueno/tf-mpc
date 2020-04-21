#!/usr/bin/env python
# coding: utf-8

import click
import numpy as np

import tfmpc.problems
import tfmpc.solvers.lqr


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
    initial_state = list(map(float, initial_state.split()))
    x0 = np.array(initial_state, dtype=np.float32)[:,np.newaxis]
    state_size = len(initial_state)

    problem = tfmpc.problems.make_lqr(state_size, action_size)
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
    initial_state = list(map(float, initial_state.split()))
    x0 = np.array(initial_state, dtype=np.float32)[:,np.newaxis]

    goal = list(map(float, goal.split()))
    g = np.array(goal, dtype=np.float32)[:,np.newaxis]

    problem = tfmpc.problems.make_lqr_linear_navigation(g, beta)
    trajectory = tfmpc.solvers.lqr.solve(problem, x0, horizon)

    print(repr(trajectory))
    print()
    print(str(trajectory))
