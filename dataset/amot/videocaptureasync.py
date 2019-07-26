#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import threading
import cv2

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.lock = threading.Lock()

    def set(self, var1, var2):
        with self.lock:
            self.cap.set(var1, var2)

    def read(self):
        with self.lock:
            ret, frame = self.cap.read()
        return ret, frame

    def open(self, src):
        self.cap.open(src)

    def isOpened(self):
        return self.cap.isOpened()

    @staticmethod
    def get_frame(src, index):
        vc = cv2.VideoCapture(src)
        vc.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = vc.read()
        vc.release()
        del vc
        return ret, frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()