#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import json
import os

class Config:
    instance = None
    def __init__(self, config_name):
        self.config = None
        self.config_name = config_name
        configure_file = os.path.join(os.path.dirname(__file__), config_name)
        with open(configure_file, 'r', encoding='utf-8') as f:
           self.config = json.load(f)
        if self.config is None:
            raise Exception("Cannot load configure files {}".format(configure_file))

    @staticmethod
    def get_configure(config_name=None):
        if Config.instance is None or Config.instance.config_name != config_name:
            if config_name is None:
                raise Exception("Please init configure specified by a name")
            Config.instance = Config(config_name)
        return Config.instance.config




