import numpy as np
import pytest
import tensorflow as tf

from tfmpc.envs.hvac import HVAC
from tfmpc.envs.navigation import Navigation
from tfmpc.envs.reservoir import Reservoir


def sample_state(env, batch_size=None):
    state_size = env.state_size
    shape = [batch_size, state_size, 1] if batch_size else [state_size, 1]
    return tf.random.normal(shape=shape, mean=10.0, dtype=tf.float32)


def sample_action(env, batch_size=None):
    action_size = env.action_size
    shape = [batch_size, action_size, 1] if batch_size else [action_size, 1]
    return tf.random.uniform(shape, dtype=tf.float32)



def sample_adjacency_matrix(n):
    adj = (tf.random.uniform(shape=[n, n], maxval=1.0) >= 0.4)
    adj = tf.logical_and(adj, tf.eye(n) <= 0)
    adj = tf.linalg.band_part(adj, 0, -1) # upper triangular
    return adj


@pytest.fixture(scope="module", name="hvac")
def hvac_fixture():
    return hvac()


def hvac():
    n_rooms = tf.random.uniform(shape=[], minval=2, maxval=6, dtype=tf.int32)

    temp_outside = tf.random.normal(shape=[n_rooms, 1], mean=6.0, dtype=tf.float32)
    temp_hall = tf.random.normal(shape=[n_rooms, 1], mean=10.0, dtype=tf.float32)

    temp_lower_bound = tf.random.normal(shape=[n_rooms, 1], mean=20.0, stddev=1.5, dtype=tf.float32)
    temp_upper_bound = temp_lower_bound + tf.random.uniform(shape=[], minval=3.5, maxval=55.5)

    R_outside = tf.random.normal(shape=[n_rooms, 1], mean=4.0, dtype=tf.float32)
    R_hall = tf.random.normal(shape=[n_rooms, 1], mean=2.0, stddev=0.5, dtype=tf.float32)
    R_wall = tf.random.normal(shape=[n_rooms, n_rooms], mean=1.5, stddev=0.1, dtype=tf.float32)
    R_wall = 1 / 2 * (R_wall + tf.transpose(R_wall))

    capacity = tf.random.normal(shape=[n_rooms, 1], mean=80.0, stddev=2.0, dtype=tf.float32)

    air_max = tf.random.normal(shape=[n_rooms, 1], mean=15.0, dtype=tf.float32)

    adj = sample_adjacency_matrix(int(n_rooms))
    adj_outside = (tf.random.normal(shape=[n_rooms, 1]) >= 0.0)
    adj_hall = (tf.random.normal(shape=[n_rooms, 1]) >= 0.0)

    return HVAC(
        temp_outside, temp_hall,
        temp_lower_bound, temp_upper_bound,
        R_outside, R_hall, R_wall,
        capacity, air_max,
        adj, adj_outside, adj_hall
    )



MAX_RES_CAP = 100


def linear_topology(n):
    row = tf.constant([1] + [0] * (n - 1), dtype=tf.float32)
    rows = [tf.roll(row, shift=i, axis=0) for i in range(1, n)]
    rows.append(tf.zeros([n,], dtype=tf.float32))
    rows = tf.stack(rows, axis=0)
    return rows


@pytest.fixture(scope="module", name="reservoir")
def reservoir_fixture():
    return reservoir()


def reservoir():
    n_reservoirs = np.random.randint(low=2, high=5)

    max_res_cap = tf.constant(
        [MAX_RES_CAP] * n_reservoirs,
        shape=[n_reservoirs, 1], dtype=tf.float32)

    lower_bound = MAX_RES_CAP * tf.random.uniform(shape=[n_reservoirs, 1],
                                                  maxval=0.5,
                                                  dtype=tf.float32)
    upper_bound = MAX_RES_CAP * tf.random.uniform(shape=[n_reservoirs, 1],
                                                  minval=0.5,
                                                  dtype=tf.float32)

    downstream = linear_topology(n_reservoirs)

    rain_mean = 0.20 * MAX_RES_CAP
    rain_variance = 0.05 * (MAX_RES_CAP ** 2)
    rain_shape = tf.constant(
        [(rain_mean ** 2) / rain_variance] * n_reservoirs,
        shape=[n_reservoirs, 1],
        dtype=tf.float32)
    rain_scale = tf.constant(
        [rain_variance / rain_mean] * n_reservoirs,
        shape=[n_reservoirs, 1],
        dtype=tf.float32)

    low_penalty = tf.constant(
        [-5.0] * n_reservoirs,
        shape=[n_reservoirs, 1],
        dtype=tf.float32)
    high_penalty = tf.constant(
        [-100.0] * n_reservoirs,
        shape=[n_reservoirs, 1],
        dtype=tf.float32)
    set_point_penalty = tf.constant(
        [-0.1] * n_reservoirs,
        shape=[n_reservoirs, 1],
        dtype=tf.float32)

    return Reservoir(
        max_res_cap, lower_bound, upper_bound,
        low_penalty, high_penalty, set_point_penalty,
        downstream,
        rain_shape, rain_scale)


@pytest.fixture(params=[1, 2], ids=["1-zone", "2-zones"], name="navigation")
def navigation_fixture(request):
    yield navigation(request.param)


def navigation_1_zone():
    return navigation(1)


def navigation_2_zones():
    return navigation(2)


def navigation(n_zones):
    goal = tf.constant([[8.0], [9.0]])
    deceleration = {
        "center": tf.random.normal(shape=[n_zones, 2, 1]),
        "decay": tf.random.uniform(shape=[n_zones,], maxval=3.0)
    }
    low = [-1.0, -1.0]
    high = [1.0, 1.0]
    return Navigation(goal, deceleration, low, high)
