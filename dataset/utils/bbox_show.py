#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import cv2

def show_bboxes_with_alpha(frame, bboxes, color=None, titles=None, time=None):
    if frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    for bbox in bboxes:
        frame = cv2.rectangle(frame,
                      tuple(bbox[:2].astype(int)),
                      tuple(bbox[-2:].astype(int)),
                      color,
                      -1)
    if time is not None:
        frame = cv2.putText(frame, str(time), (25, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255, 125), 2)

    if titles is not None:
        for i, title in enumerate(titles):
            frame = cv2.putText(frame,
                        title,
                        tuple(bboxes[i, :][:2].astype(int)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        color,
                        2)

    return frame


def show_bboxes(frame, bboxes, color=None, titles=None, alpha=0.3, time=None):

    overlay = frame.copy()
    output = frame.copy()

    for bbox in bboxes:
        cv2.rectangle(overlay,
                      tuple(bbox[:2].astype(int)),
                      tuple(bbox[-2:].astype(int)),
                      color,
                      -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    if time is not None:
        cv2.putText(output, str(time), (25, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if titles is not None:
        for i, title in enumerate(titles):
            cv2.putText(output,
                        title,
                        tuple(bboxes[i, :][:2].astype(int)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        color,
                        2)
    return output
