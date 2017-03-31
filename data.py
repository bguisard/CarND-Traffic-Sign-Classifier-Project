import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt


def convert_to_YUV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def plot_img(img, figsize=(12, 1)):
    implot = plt.figure(*figsize)
    implot.grid(False)
    return implot.imshow(img)
