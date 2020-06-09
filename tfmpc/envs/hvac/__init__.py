import gym
import numpy as np
import tensorflow as tf

from tfmpc.envs.diffenv import DiffEnv


class HVAC(DiffEnv):

    CAP_AIR = tf.constant(1.006, dtype=tf.float32)
    COST_AIR = tf.constant(1.0, dtype=tf.float32)
    TEMP_AIR = tf.constant(40.0, dtype=tf.float32)
    TIME_DELTA = tf.constant(1.0, dtype=tf.float32)
    PENALTY = tf.constant(20_000, dtype=tf.float32)
    SET_POINT_PENALTY = tf.constant(10.0, dtype=tf.float32)

    def __init__(self,
                 temp_outside,
                 temp_hall,
                 temp_lower_bound,
                 temp_upper_bound,
                 R_outside,
                 R_hall,
                 R_wall,
                 capacity,
                 air_max,
                 adj,
                 adj_outside,
                 adj_hall):

        # temperature outside and hall
        self.temp_outside = temp_outside
        self.temp_hall = temp_hall

        # temperature limits
        self.temp_lower_bound = temp_lower_bound
        self.temp_upper_bound = temp_upper_bound

        # thermal conductance
        self.R_outside = R_outside
        self.R_hall = R_hall
        self.R_wall = R_wall

        # heat capacity
        self.capacity = capacity

        # air maximum volume
        self.air_max = air_max

        # room adjacency matrix
        self.adj = adj
        self.adj_outside = adj_outside
        self.adj_hall = adj_hall

        self.obs_space = gym.spaces.Box(
            shape=[self.state_size, 1], low=-np.inf, high=np.inf)

        self.action_space = gym.spaces.Box(
            shape=[self.action_size, 1], low=0.0, high=1.0)

    @property
    def state_size(self):
        return len(self.temp_lower_bound)

    @property
    def action_size(self):
        return self.state_size

    @tf.function
    def transition(self, state, action, batch=False):
        temp = state
        air = action * self.air_max

        heating = air * self.CAP_AIR * (self.TEMP_AIR - temp)

        conduction_between_rooms = self._conduction_between_rooms(temp)
        conduction_with_outside = self._conduction_with_outside(temp)
        conduction_with_hall = self._conduction_with_hall(temp)

        temp_ = (
            temp
            + self.TIME_DELTA / self.capacity * (
                heating
                + conduction_between_rooms
                + conduction_with_outside
                + conduction_with_hall
            )
        )
        return temp_

    @tf.function
    def cost(self, state, action, batch=False):
        temp = state
        air = action * self.air_max

        air_cost = self.COST_AIR * air
        out_of_bounds_penalty = self.PENALTY * (
            tf.maximum(0.0, self.temp_lower_bound - temp)
            + tf.maximum(0.0, temp - self.temp_upper_bound)
        )
        set_point_penalty = (
            self.SET_POINT_PENALTY
            * tf.abs(
                (self.temp_lower_bound + self.temp_upper_bound) / 2 - temp)
        )

        return tf.reduce_sum(
            tf.squeeze(air_cost + out_of_bounds_penalty + set_point_penalty, axis=-1),
            axis=-1
        )

    @tf.function
    def final_cost(self, state):
        temp = state

        out_of_bounds_penalty = self.PENALTY * (
            tf.maximum(0.0, self.temp_lower_bound - temp)
            + tf.maximum(0.0, temp - self.temp_upper_bound)
        )
        set_point_penalty = (
            self.SET_POINT_PENALTY
            * tf.abs(
                (self.temp_lower_bound + self.temp_upper_bound) / 2 - temp)
        )

        return tf.reduce_sum(
            tf.squeeze(out_of_bounds_penalty + set_point_penalty, axis=-1),
            axis=-1
        )

    @tf.function
    def _conduction_between_rooms(self, temp, batch=False):
        adj = tf.logical_or(self.adj, tf.transpose(self.adj))
        adj = tf.cast(adj, tf.float32)
        return tf.reduce_sum(
            - adj / self.R_wall * (temp - tf.linalg.matrix_transpose(temp)),
            axis=-1,
            keepdims=True
        )

    @tf.function
    def _conduction_with_outside(self, temp):
        adj_outside = tf.cast(self.adj_outside, tf.float32)
        return adj_outside / self.R_outside * (self.temp_outside - temp)

    @tf.function
    def _conduction_with_hall(self, temp):
        adj_hall = tf.cast(self.adj_hall, tf.float32)
        return adj_hall / self.R_hall * (self.temp_hall - temp)

    def __repr__(self):
        return f"HVAC({self.state_size})"

    def __str__(self):
        bounds = [
            f"[{float(low):.3f}, {float(high):.3f}]"
            for low, high in zip(
                    tf.squeeze(self.temp_lower_bound),
                    tf.squeeze(self.temp_upper_bound))]
        bounds = f"[{', '.join(bounds)}]"

        R = [
            f"[outside={float(o):.3f}, hall={float(h):.3f}]"
            for o, h in zip(
                    tf.squeeze(self.R_outside),
                    tf.squeeze(self.R_hall))]
        R = "\n".join(R)

        R_wall = self.R_wall.numpy()

        capacity = [f"{c:.3f}" for c in tf.squeeze(self.capacity)]
        capacity = f"[{', '.join(capacity)}]"

        air_max = [f"{c:.3f}" for c in tf.squeeze(self.air_max)]
        air_max = f"[{', '.join(air_max)}]"

        adj = self.adj.numpy()
        adj_outside = tf.squeeze(self.adj_outside).numpy().tolist()
        adj_hall = tf.squeeze(self.adj_hall).numpy().tolist()

        temp_outside = [f"{c:.3f}" for c in tf.squeeze(self.temp_outside)]
        temp_outside = f"[{', '.join(temp_outside)}]"

        temp_hall = [f"{c:.3f}" for c in tf.squeeze(self.temp_hall)]
        temp_hall = f"[{', '.join(temp_hall)}]"

        return f"HVAC(\ntemp_bounds={bounds},\nR=\n{R},\nR_wall=\n{R_wall},\ncapacity={capacity},\nair_max={air_max},\nadj=\n{adj},\nadj_outside={adj_outside},\nadj_hall={adj_hall},\ntemp_outside={temp_outside},\ntemp_hall={temp_hall}\n)"
    @classmethod
    def load(cls, config):
        for key, value in config.items():
            if key.startswith("adj"):
                config[key] = tf.constant(value, dtype=tf.bool)
            else:
                config[key] = tf.constant(value, dtype=tf.float32)
        return cls(**config)
