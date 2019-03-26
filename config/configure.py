import json
import os

def get_config(configure_name):
    config = None
    configure_file = os.path.join(os.path.dirname(__file__), configure_name)
    with open(configure_file, 'r', encoding='utf-8') as f:
       config = json.load(f)
    if config is None:
        raise Exception("Cannot load configure files {}".format(configure_file))

    return config



