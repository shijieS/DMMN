from config import config
import torch
from torchvision import transforms
import cv2
import numpy as np
from numpy import random
from motion_model import MotionModel

cfg = config[config["phase"]]


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union # [A,B]


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

    def __call__(self, i1):
        for t in self.transforms:
            i1 = t(i1)
        return i1


class ToFloat(object):
    def __call__(self, items):
        if items is None or items[3] is None:
            return None

        for i in range(len(items[3])):
            items[3][i] = items[3][i].astype(np.float32)

        return items


class ToAbsoluteCoords(object):
    def __call__(self, items):
        if items is None:
            return None

        h, w, c = items[3][0].shape
        items[2][:, :, [0, 2]] *= w
        items[2][:, :, [1, 3]] *= h

        return items


class ToPercentCoords(object):
    def __call__(self, items):
        if items is None:
            return None
        h, w, c = items[3][0].shape
        items[2][:, :, [0, 2]] /= w
        items[2][:, :, [1, 3]] /= h

        return items


class ConvertColor(object):
    def __init__(self, current="BGR", transform="HSV"):
        self.current = current
        self.transform = transform

    def __call__(self, images):
        for i in range(len(images)):
            if self.current == 'BGR' and self.transform == 'HSV':
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
            elif self.current == 'HSV' and self.transform == 'BGR':
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2BGR)
            else:
                raise NotImplementedError

        return images


class RandomContrast():
    def __init__(self, lower=cfg["contrast_lower"], upper=cfg["contrast_upper"]):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, images):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            for i in range(len(images)):
                images[i] *= alpha

        return images


class RandomSaturation(object):
    def __init__(self, lower=cfg["saturation_lower"], upper=cfg["saturation_upper"]):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, images):
        if random.randint(2):
            weight = random.uniform(self.lower, self.upper)
            for i in range(len(images)):
                images[i][:, :, 1] *= weight

        return images


class RandomHue(object):
    def __init__(self, delta=cfg["hue_delta"]):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, images):
        if random.randint(2):
            bias = random.uniform(-self.delta, self.delta)
            for i in range(len(images)):

                images[i][:, :, 0] += bias
                images[i][:, :, 0][images[i][:, :, 0] > 360.0] -= 360.0
                images[i][:, :, 0][images[i][:, :, 0] < 0.0] += 360.0

        return images


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, images):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            for i in range(len(images)):
                images[i] = images[i][:, :, swap]

        return images


class RandomBrightness(object):
    def __init__(self, delta=cfg["brightness_delta"]):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, images):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            for i in range(len(images)):
                images[i] += delta
        return images


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
        if items is None:
            return None

        items[3] = self.rand_brightness(items[3])
        if random.randint(2):
            distort = Compose2(self.pd[:-1])
        else:
            distort = Compose2(self.pd[1:])

        items[3] = distort(items[3])
        items[3] = self.rand_light_noise(items[3])
        return items



class CalculateParameters(object):
    def __call__(self, items):
        if items is None:
            return None

        # re-calcuate the curve parameters
        parameters, p_e, p_c = MotionModel.get_parameters(
            bboxes=items[2], times=items[4],
            invalid_node_rate=config['min_valid_node_rate'])

        if sum(p_c) == 0:
            return None

        items += [parameters, p_e, p_c]

        items[1] = items[1][p_c]
        items[2] = items[2][:, p_c, :]
        items[5] = items[5][p_c, : , :]
        items[6] = items[6][:, p_c]
        items[7] = items[7][p_c]

        return items


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, items, max_expand_ratio=cfg["max_expand_ratio"]):
        if items is None:
            return None

        if random.randint(2):
            return items

        height, width, depth = items[3][0].shape
        ratio = random.uniform(1, cfg["max_expand_ratio"])
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        mask = np.sum(items[2], axis=2) == 0
        items[2][:, :, [0, 2]] /= ratio
        items[2][:, :, [0, 2]] += left/(width*ratio)
        items[2][:, :, [1, 3]] /= ratio
        items[2][:, :, [1, 3]] += top/(height*ratio)
        items[2][mask] = 0

        for i in range(len(items[3])):
            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=items[3][i].dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height), int(left):int(left + width)] = items[3][i]
            items[3][i] = expand_image

        return items

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )


    def __call__(self, items):
        if items is None:
            return None

        all_images = items[3]
        all_bboxes = items[2]

        height, width, _ = all_images[0].shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return items

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                i = 0
                image = all_images[i]
                boxes = all_bboxes[i, :, :]
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

        return current_image, current_boxes, current_labels

# TODO: Fix
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


class Resize(object):
    def __init__(self, size=config["frame_size"]):
        self.size = size

    def __call__(self, items):
        if items is None:
            return None

        for i in range(len(items[3])):
            items[3][i] = cv2.resize(items[3][i], (self.size, self.size))
        return items


class SubtractMeans(object):
    def __init__(self, mean=config["pixel_mean"]):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, items):
        if items is None:
            return None
        for i in range(len(items[3])):
            items[3][i] = items[3][i].astype(np.float32)
            items[3][i] -= self.mean

        return items


class AddMeans(object):
    def __init__(self, mean=config["pixel_mean"]):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, items):
        if items is None:
            return None
        for i in range(len(items[3])):
            items[3][i] = items[3][i].astype(np.float32)
            items[3][i] += self.mean

        return items

class ToTensor(object):
    def __call__(self, items):
        if items is None:
            return None

        items[0] = torch.from_numpy(items[0])
        items[1] = torch.from_numpy(items[1])
        items[2] = torch.from_numpy(items[2])
        items[3] = torch.stack([torch.from_numpy(i) for i in items[3]], 0)
        items[4] = torch.from_numpy(items[4])
        items[5] = torch.from_numpy(items[5])
        items[6] = torch.from_numpy(items[6].astype(np.uint8))
        items[7] = torch.from_numpy(items[7].astype(np.uint8))

        return items

class ToNumpy(object):

    def image_tensor_2_cv(self, img):
        img = img.numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img



    def __call__(self, items):
        if items is None:
            return None

        items[0] = items[0].numpy()
        items[1] = items[1].numpy()
        items[2] = items[2].numpy()
        items[3] = [self.image_tensor_2_cv(i) for i in items[3]]
        items[4] = items[4].numpy()
        items[5] = items[5].numpy()
        items[6] = items[6].numpy().astype(bool)
        items[7] = items[7].numpy().astype(bool)

        return items

#TODO: RandomCrop
class Transforms(object):
    def __init__(self, size=config["frame_size"], mean=(104, 117, 123)):
        self.mean = mean
        self.size = size

        self.augument = Compose([
            ToFloat(),
            PhotometricDistort(),
            Expand(self.mean),
            # RandomSampleCrop(),
            # RandomMirror(),
            Resize(self.size),
            SubtractMeans(self.mean),
            CalculateParameters(),     # re-calculate the parameters of rectangles
            ToTensor()
        ])

    def __call__(self, items):
        return self.augument(items)

class TransformsTest(object):
    def __init__(self, size=config["frame_size"], mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augument = Compose([
            ToFloat(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            CalculateParameters(),  # re-calculate the parameters of rectangles
            ToTensor()
        ])

    def __call__(self, items):
        return self.augument(items)


class TransformReader(object):

    def __init__(self):
        self.augument = Compose([
            ToFloat(),
            ToAbsoluteCoords(),
            ToPercentCoords(),
            PhotometricDistort(),
            Resize(300),
            SubtractMeans(config["pixel_mean"]),
            AddMeans(config["pixel_mean"]),
            Expand(config["pixel_mean"]),
            CalculateParameters(),
            ToTensor(),
            ToNumpy(),
        ])

    def __call__(self, items):
        return self.augument(items)

