import wandb


def setup(filename, filemode="w"):
    pass


def log(timestep, row):
    wandb.log(row, step=timestep)


def log_transition(timestep, state, action, cost, next_state, done, info):
    del next_state
    del done

    fluent_dict = {}
    for fluent_type, fluent in [("x", state), ("u", action)]:
        for i, value in enumerate(fluent):
            fluent_dict[f"{fluent_type}[{i}]"] = value

    total_cost = info["total_cost"]

    wandb.log(
        {
            **fluent_dict,
            "cost": cost,
            "reward": -cost,
            "cum_total_cost": total_cost,
            "cum_total_reward": -total_cost
        },
        step=timestep
    )


def summary(trajectory, uptime):
    total_cost = trajectory.total_cost
    wandb.run.summary["total_reward"] = -total_cost
    wandb.run.summary["total_cost"] = total_cost
    wandb.run.summary["uptime"] = uptime
