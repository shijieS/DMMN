from pprint import pprint
from.config import get_config
from dataset.utils import MotionModel

configure_name = 'config_train_ua.json'


config = get_config(configure_name)

# init motion parameter number
config['num_motion_model_param'] = MotionModel.get_num_parameter()

# init end iterations
config["iteration"] = config["start_iter"] + config["epoch_num"]*config["epoch_size"]

print('loading configure: ' + configure_name + "========")
pprint(config)
