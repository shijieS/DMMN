from pprint import pprint
from.config import get_config

configure_name = 'config_mot17.txt'


config = get_config(configure_name)

print('loading configure: ' + configure_name + "========")
pprint(config)
