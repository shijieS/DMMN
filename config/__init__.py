from pprint import pprint
from.config import get_config
from dataset.MotionModel import MotionModel

configure_name = 'config_train_ua_ssm.json'

config = get_config(configure_name)

# init motion parameter number
config['num_motion_model_param'] = MotionModel.get_num_parameter()

print('loading configure: ' + configure_name + "========")
pprint(config)
