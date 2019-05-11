#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

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
