import yaml
from types import SimpleNamespace
from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def parse_config(config):
    if type(config) is list:
        return list(map(parse_config, config))
    elif type(config) is dict:
        sns = SimpleNamespace()
        for key, value in config.items():
            setattr(sns, key, parse_config(value))
        return sns
    else:
        return config

