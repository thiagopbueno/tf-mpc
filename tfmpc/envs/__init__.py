import importlib


def make_env(env, config=None):
    module = importlib.import_module(f"tfmpc.envs.{env}")
    cls_name = env.capitalize()
    return getattr(module, cls_name).load(config)
