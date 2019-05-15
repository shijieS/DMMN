#  #!/usr/bin/env python
#   Copyright (c) 2019. ShiJie Sun at the Chang'an University
#   This work is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.
#   For a copy, see <http://creativecommons.org/licenses/by-nc-sa/3.0/>.
#   Author: shijie Sun
#   Email: shijieSun@chd.edu.cn
#   Github: www.github.com/shijieS
#

import cv2
import numpy as np

class DrawBoxes:
    @staticmethod
    def cv_draw_one_box(frame,
                        box,
                        color,
                        content_color=None,
                        alpha=0.5,
                        text="",
                        font_color=None,
                        with_border=True,
                        border_color=None):
        """
        Draw a box on a frame
        :param frame:
        :param box:
        :param color:
        :param alpha:
        :param text:
        :param font_color:
        :param with_border:
        :param border_color:
        :return:
        """
        # draw box content
        if content_color is None:
            content_color = color

        if text == "NO":
            return frame

        (l, t, r, b) = tuple([int(b) for b in box])
        roi = frame[t:b, l:r]
        black_box = np.zeros_like(roi)
        black_box[:, :, :] = content_color
        cv2.addWeighted(roi, alpha, black_box, 1-alpha, 0, roi)

        # draw border
        if with_border:
            if border_color is None:
                border_color = color
            print(l, t, r, b)
            if l + r >  0 and r-l < 100000 and r-l > -100000 and b - l < 100000 and b - l > -100000:
                cv2.rectangle(frame, (l, t), (r, b), border_color, 1)

        # put text
        if font_color is None:
            font_color = color

        if l + r > 0 and r - l > -10000 and r - l < 10000 and b - l < 100000 and b - l > -100000:
            cv2.putText(frame, text, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.3, font_color)

        return frame

    @staticmethod
    def cv_draw_mult_boxes(frame, boxes, colors=None, texts=None):
        """
        Draw multiple boxes on one frame
        :param frame: the frame to be drawn
        :param boxes: all the boxes, whoes shape is [n, 4]
        :param color: current boxes' color
        :return:
        """
        boxes_len = len(boxes)
        if colors is None:
            colors = [DrawBoxes.get_random_color(i) for i in range(boxes_len)]

        if texts is None:
            texts = ["" for _ in range(boxes_len)]

        for box, color, text in zip(boxes, colors, texts):
            frame = DrawBoxes.cv_draw_one_box(frame, box, color, text=text)

        return frame

    @staticmethod
    def cv_draw_mult_boxes_with_track(frame, boxes, index, colors=None, texts=None):
        """
        Draw multiple boxes with its track
        :param frame: the frame to be drawn
        :param boxes: the set of boxes, with the shape [m, n, 4] where, m is the frame number, n is the number of box in each frame.
        :param index:
        :return:
        """
        frame_num, boxes_num, _ = boxes.shape
        # get each boxes' color`
        if colors is None:
            colors = DrawBoxes.get_random_colors(boxes_num)

        if colors is None:
            texts = ["" for _ in range(boxes_num)]

        # draw frame's boxes at the specified index
        DrawBoxes.cv_draw_mult_boxes(frame, boxes[index, :, :], colors, texts)

        # draw tracks
        for box_index in range(boxes_num):
            color = colors[box_index]
            DrawBoxes.cv_draw_track_(frame, boxes[:, box_index, :], color)


    @staticmethod
    def cv_draw_one_box_center(frame, box, color=None, radius=1, thickness=1):
        """
        Draw the center of the box on the frame
        :param frame: frame to be drawn
        :param box: box with the shape (l, t , r, b)
        :param color: color, i.e. (0, 255, 0)
        :param radius: the center point's radius
        :param thickness: the border of center points' thickness
        """
        if color is None:
            color = DrawBoxes.get_random_color()

        (l, t, r, b) = tuple([b for b in box])
        x_c = (l + r) // 2
        y_c = int(b)
        cv2.circle(frame,  (x_c, y_c), radius, color, thickness)

    @staticmethod
    def cv_draw_mult_box_center(frame, boxes, colors, radius=1, thickness=1):
        if colors is None:
            colors = DrawBoxes.get_random_colors(len(boxes))
        for b, c in zip(boxes, colors):
            DrawBoxes.cv_draw_one_box_center(frame, b, c, radius, thickness)


    @staticmethod
    def cv_draw_track_(frame, boxes, color=None, thickness=1):
        if color is None:
            color = DrawBoxes.get_random_color()

        points = []
        for b in boxes:
            (l, t, r, b) = tuple([i for i in b])
            if l + r == 0 or r - l < -10000 or r - l > 10000 or b - l > 100000 or b - l < -100000:
                continue
            x = int((l + r) // 2)
            y = int(b)
            cv2.circle(frame, (x, y), 1, color, thickness*2)
            # if x == 0 and y == 0:
            #     continue
            points += [(x, y)]

        if len(points) > 1:
            for i in range(len(points)-1):
                cv2.line(frame, points[i], points[i+1], color, thickness)


    @staticmethod
    def get_random_color(seed=None):
        """
        Get the random color.
        :param seed: if seed is not None, then seed the random value
        :return:
        """
        if seed is not None:
            np.random.seed(seed)

        return tuple([np.random.randint(0, 255) for i in range(3)])


    @staticmethod
    def get_random_colors(num, is_seed=True):
        """
        Get a set of random color
        :param num: the number of color
        :param is_seed: is the random seeded
        :return: a list of colors, i.e. [(255, 0, 0), (0, 255, 0)]
        """
        if is_seed:
            colors = [DrawBoxes.get_random_color(i) for i in range(num)]
        else:
            colors = [DrawBoxes.get_random_color() for _ in range(num)]
        return colors
