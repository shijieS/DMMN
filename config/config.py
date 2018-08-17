import json
import os

def get_config(configure_name):
    configure_file = os.path.join(os.path.dirname(__file__), configure_name)
    with open(configure_file, 'r', encoding='utf-8') as f:
       config = json.load(f)


