import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, transform
import bcolz
import random
import itertools


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def plot_img(img, figsize=(2, 2)):
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            # print ("Flattening grayscale img")
            img = img[:, :, 0]

    implot = plt.figure(figsize=figsize)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)


def YUV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def YUV2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def YCC(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def YCC2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


def Histeq(img):
    yuv = YUV(img)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return YUV2RGB(yuv)


def Adapthisteq(img):
    return exposure.equalize_adapthist(img)


def Motionblur(img, kernel_sz=3):
    imshape = img.shape
    kernel_mb = np.zeros((kernel_sz, kernel_sz))
    kernel_mb[int((kernel_sz - 1) / 2), :] = np.ones(kernel_sz)
    kernel_mb = kernel_mb / kernel_sz
    blur = cv2.filter2D(img, -1, kernel_mb)
    return blur.reshape(*imshape)


def Rotate(img, angle_limit=30.):
    return transform.rotate(img, random.uniform(-angle_limit, angle_limit), mode='edge')


def Affine(img, scale_limit=0.1, angle_limit=15., shear_limit=10., trans_limit=2):
    h, w = img.shape[:2]
    centering = np.array((h, w)) / 2. - 0.5
    scale = random.uniform(1 - scale_limit, 1 + scale_limit)
    angle = np.deg2rad(random.uniform(-angle_limit, angle_limit))
    shear = np.deg2rad(random.uniform(-shear_limit, shear_limit))
    trans_x = random.uniform(-trans_limit, trans_limit)
    trans_y = random.uniform(-trans_limit, trans_limit)

    center = transform.SimilarityTransform(translation=-centering)
    tform = transform.AffineTransform(scale=(scale, scale),
                                      rotation=angle,
                                      shear=shear,
                                      translation=(trans_x, trans_y))
    recenter = transform.SimilarityTransform(translation=centering)

    return transform.warp(img, (center + (tform + recenter)).inverse, mode='edge')


def Augmentation(img, p_blur=0.2, p_affine=1):
    p = random.uniform(0., 1.)
    if p_affine >= p:
        img = Affine(img)

    p = random.uniform(0., 1.)
    if p_blur >= p:
        img = Motionblur(img)

    return img


def img_preprocess(img):
    pp_img = gray(img)
    pp_img = Adapthisteq(pp_img).astype(np.float32)
    pp_img = pp_img.reshape(pp_img.shape + (1,))
    return pp_img


def preprocess_array(X):
    Xp = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1)).astype(np.float32)
    for i in range(X.shape[0]):
        Xp[i] = img_preprocess(X[i])
    return Xp


def Augment_dataset(X, y=None, augs=19, preprocessed=True):
    X_aug = np.zeros((X.shape[0] * (augs + 1), X.shape[1], X.shape[2], X.shape[3]))
    counter = 0
    if y is not None:
        assert len(X) == len(y)
        y_aug = np.zeros(y.shape[0] * (augs + 1))

    for i in range(X.shape[0]):

        if not preprocessed:
            X[i] = img_preprocess(X[i])

        X_aug[counter] = X[i]

        if y is not None:
            y_aug[counter] = y[i]

        counter += 1

        for n in range(augs):
            X_aug[counter] = Augmentation(X[i])

            if y is not None:
                y_aug[counter] = y[i]

            counter += 1
    return X_aug, y_aug


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
