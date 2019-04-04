from pprint import pprint
from.configure import Config
# from motion_model.motion_model_quadratic import motion_model

# configure_name = 'config_test_ua_lab_debug.json'
configure_name = 'config_test_gpu4_debug.json'

config = Config.get_configure(configure_name)
cfg = config[config["phase"]]
# init motion parameter number
# config['num_motion_model_param'] = motion_model.get_num_parameter()

print('loading configure: ' + configure_name + "========")
pprint(config)
