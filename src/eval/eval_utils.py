import cv2
import time
import torch
import random
import numpy as np


def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp