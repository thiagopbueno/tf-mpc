import logging


def setup(filename, filemode="w"):
    logging.basicConfig(filename=filename, filemode=filemode, level=logging.DEBUG)


def log(timestep, row):
    logging.info(f">> timestep = {timestep} : {row}")


def log_transition(timestep, state, action, cost, next_state, done, info):
    logging.info(f">> timestep = {timestep}")
    logging.info(f" state = {state.numpy().tolist()}")
    logging.info(f" action = {action.numpy().tolist()}")
    logging.info(f" cost = {cost}")
    logging.info(f" cumulative_total_cost = {info['total_cost']}")


def summary(trajectory, uptime):
    total_cost = trajectory.total_cost
    logging.info(f">> total_cost = {total_cost}")
    logging.info(f">> uptime = {uptime}")
