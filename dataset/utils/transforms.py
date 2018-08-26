from config import config
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from ..MotionModel import MotionModel


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, items):
        for t in self.transforms:
            items = t(items)
        return items


class Compose2(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, i1, i2):
        for t in self.transforms:
            i1, i2 = t(i1, i2)
        return i1, i2


class ConvertFromInts(object):
    def __call__(self, items):
        for i in range(len(items[0])):
            items[0][i] = items[0][i].astype(np.float32)
            items[6][i] = items[6][i].astype(np.float32)

        return items


class ToAbsoluteCoords(object):
    def __call__(self, items):
        h, w, c = items[0][0].shape
        items[1][[0, 2], :] *= w
        items[1][[1, 3], :] *= h

        items[6][[0, 2], :] *= w
        items[6][[1, 3], :] *= h

        return items


class ToPercentCoords(object):
    def __call__(self, items):
        h, w, c = items[0][0].shape
        items[1][[0, 2], :] /= w
        items[1][[1, 3], :] /= h

        items[7][[0, 2], :] /= w
        items[7][[1, 3], :] /= h

        return items


class ConvertColor(object):
    def __init__(self, current="BGR", transform="HSV"):
        self.current = current
        self.transform = transform

    def __call__(self, images_1, images_2):
        for i in range(len(images_1)):
            (i_1, i_2) = (images_1[i], images_2[2])
            if self.current == 'BGR' and self.transform == 'HSV':
                i_1 = cv2.cvtColor(i_1, cv2.COLOR_BGR2HSV)
                i_2 = cv2.cvtColor(i_2, cv2.COLOR_BGR2HSV)
            elif self.current == 'HSV' and self.transform == 'BGR':
                i_1 = cv2.cvtColor(i_1, cv2.COLOR_HSV2BGR)
                i_2 = cv2.cvtColor(i_2, cv2.COLOR_HSV2BGR)
            else:
                raise NotImplementedError

        return images_1, images_2

class RandomContrast():
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, images_1, images_2):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            for i in range(len(images_1)):
                images_1[i] *= alpha
                images_2[i] *= alpha

        return images_1, images_2


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, images_1, images_2):
        if random.randint(2):
            weight = random.uniform(self.lower, self.upper)
            for i in range(len(images_1)):
                images_1[i][:, :, 1] *= weight
                images_2[i][:, :, 1] *= weight

        return images_1, images_2


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, images_1, images_2):
        if random.randint(2):
            bias = random.uniform(-self.delta, self.delta)
            for i in range(len(images_1)):
                (i1, i2) = (images_1[i], images_2[i])

                i1[:, :, 0] += bias
                i1[:, :, 0][i1[:, :, 0] > 360.0] -= 360.0
                i1[:, :, 0][i1[:, :, 0] < 0.0] += 360.0

                i2[:, :, 0] += bias
                i2[:, :, 0][i2[:, :, 0] > 360.0] -= 360.0
                i2[:, :, 0][i2[:, :, 0] < 0.0] += 360.0

        return images_1, images_2


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, images_1, images_2):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            for i in range(len(images_1)):
                i1, i2 = images_1[i], images_2[i]
                i1 = i1[:, :, swap]
                i2 = i2[:, :, swap]

        return images_1, images_2


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, images_1, images_2):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            for i in range(len(images_1)):
                (i1, i2) = (images_1[i], images_2[i])
                i1 += delta
                i2 += delta
        return images_1, images_2


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, items):
        items[0], items[6] = self.rand_brightness(items[0], items[6])
        if random.randint(2):
            distort = Compose2(self.pd[:-1])
        else:
            distort = Compose2(self.pd[1:])

        items[0], items[6] = distort(items[0], items[6])
        items[0], items[6] = self.rand_light_noise(items[0], items[6])
        return items


class RecalculateParameters(object):
    def __call__(self, items):
        # re-calcuate the curve parameters
        items[2], items[3] = MotionModel.get_parameters(
            bboxes=items[1], times=items[4],
            invalid_node_rate=config['min_valid_node_rate'])
        items[8], items[9] = MotionModel.get_parameters(
            bboxes=items[7], times=items[10],
            invalid_node_rate=config['min_valid_node_rate'])
        return items


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, items):
        if random.randint(2):
            return items

        height, width, depth = items[0][0].shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        for i in range(len(items[0])):
            expand_image1 = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=items[0][i].dtype)
            expand_image1[:, :, :] = self.mean
            expand_image1[int(top):int(top + height),
            int(left):int(left + width)] = items[0][i]
            items[0][i] = expand_image1
            items[1][i, :][:, :2] += (int(left), int(top))
            items[1][i, :][:, 2:] += (int(left), int(top))

            expand_image2 = np.zeros(
                (int(height * ratio), int(width * ratio), depth),
                dtype=items[6][i].dtype)
            expand_image2[:, :, :] = self.mean
            expand_image2[int(top):int(top + height),
            int(left):int(left + width)] = items[6][i]
            items[6][i] = expand_image2
            items[7][i, :][:, :2] += (int(left), int(top))
            items[7][i, :][:, 2:] += (int(left), int(top))

        return items

# class RandomSampleCrop(object):
#     def __init__(self):
#         self.sample_options = (
#             # using entire original input image
#             None,
#             # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
#             (0.1, None),
#             (0.3, None),
#             (0.7, None),
#             (0.9, None),
#             # randomly sample a patch
#             (None, None)
#         )
#
#     def process_one_item(self, images, boxes):
#         pass
#
#     def __call__(self, items):
#         pass


class RandomMirror(object):
    def __call__(self, items):
        _, width, _ = items[0][0].shape
        if random.randint(2):
            for i in range(len(items[0])):
                items[0][i] = items[0][i][:, ::-1]
                items[1][i, :][:, 0::2] = width - items[1][i, :][:, 2::-2]

                items[6][i] = items[6][i][:, ::-1]
                items[7][i, :][:, 0::2] = width - items[7][i, :][:, 2::-2]
        # reset the mask region
        items[1] *= items[3][:, :, None]
        items[7] *= items[9][:, :, None]
        return items

class ToPercentCoords(object):
    def __call__(self, items):
        height, width, channels = items[0][0].shape
        for b1, b2 in zip(items[1], items[6]):
            b1[:, 0] /= width
            b1[:, 2] /= width
            b1[:, 1] /= height
            b1[:, 3] /= height

            b2[:, 0] /= width
            b2[:, 2] /= width
            b2[:, 1] /= height
            b2[:, 3] /= height

        return items


class Resize(object):
    def __init__(self, size=config["frame_size"]):
        self.size = size

    def __call__(self, items):
        for i in range(len(items[0])):
            items[0][i] = cv2.resize(items[0][i], (self.size, self.size))
            items[6][i] = cv2.resize(items[6][i], (self.size, self.size))
        return items


class SubtractMeans(object):
    def __init__(self, mean=config["pixel_mean"]):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, items):
        for i in range(len(items[0])):
            items[0][i] = items[0][i].astype(np.float32)
            items[0][i] -= self.mean

            items[6][i] = items[6][i].astype(np.float32)
            items[6][i] -= self.mean

        return items

class ToTensor(object):
    def __call__(self, items):
        i1 = [torch.from_numpy(i).permute(2, 0, 1) for i in items[0]]
        i1 = torch.stack(i1, 0).transpose(1, 0)
        b1 = torch.from_numpy(items[1])
        m1 = torch.from_numpy(items[2])
        p_m1 = torch.from_numpy(items[3])
        t1 = torch.from_numpy(items[4])
        p_c1 = torch.from_numpy(items[5])

        i2 = [torch.from_numpy(i).permute(2, 0, 1) for i in items[6]]
        i2 = torch.stack(i2, 0).transpose(1, 0)
        b2 = torch.from_numpy(items[7])
        m2 = torch.from_numpy(items[8])
        p_m2 = torch.from_numpy(items[9])
        t2 = torch.from_numpy(items[10])
        p_c2 = torch.from_numpy(items[11])

        s = torch.from_numpy(items[12])

        return i1, b1, m1, p_m1, t1, p_c1, \
               i2, b2, m2, p_m2, t2, p_c2, \
               s

class Transforms(object):
    def __init__(self, size=config["frame_size"], mean=(104, 117, 123)):
        self.mean = mean
        self.size = size

        self.augument = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            # RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            RecalculateParameters(),     # re-calculate the parameters of rectangles
            ToTensor()
        ])

    def __call__(self, items):
        return self.augument(items)
