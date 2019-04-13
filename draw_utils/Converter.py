from config import config, cfg
import cv2
import torch
import numpy as np

class TypeConverter:
    @staticmethod
    def tensor_2_numpy_gpu(data):
        return data.cpu().numpy()

    @staticmethod
    def tensor_2_numpy(data):
        return data.numpy()

    @staticmethod
    def image_tensor_2_cv(data):
        img = TypeConverter.tensor_2_numpy(data)
        img = img.transpose([1, 2, 0])+config['pixel_mean']
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def image_tensor_2_cv_gpu(data):
        return TypeConverter.image_tensor_2_cv(data.cpu())
