import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError("config.yaml 为空")
    return data
