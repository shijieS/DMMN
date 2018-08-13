import json
import os
from pprint import pprint
import sys

selected_file_name = 'config_mot17.txt'
configure_file = os.path.join(os.path.dirname(__file__), selected_file_name)
with open(configure_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

print('loading configure: ' + selected_file_name + "========>")
pprint(config)


