# -*- coding: utf-8 -*-
"""BSRGAN degradation pipeline.

Adapted from official BSRGAN:
Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (ICCV 2021)
Kai Zhang (cskaizhang@gmail.com)
https://github.com/cszn/BSRGAN
"""

import random
import cv2
import numpy as np
import scipy.stats as ss
import torch
from scipy import ndimage
from scipy.interpolate import interp2d
from scipy.linalg import orth

from basicsr.utils.matlab_functions import imresize


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())


def uint2single(img):
    return np.float32(img / 255.0)


def imresize_np(img, scale, antialiasing=True):
    return imresize(img, scale=scale, antialiasing=antialiasing)


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
    k = k / np.sum(k)
    return k


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1.0, 0.0]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    return gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std + 1e-8)
    h = np.exp(arg)
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def shift_pixel(x, sf, upper_left=True):
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    elif x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)
    return x


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def add_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf
    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = anisotropic_Gaussian(ksize=2 * random.randint(2, 11) + 3, theta=random.random() * np.pi, l1=l1, l2=l2)
    else:
        k = fspecial_gaussian(2 * random.randint(2, 11) + 3, wd * random.random())
    img = ndimage.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
    return img


def add_resize(img, sf=4):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5 / sf, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (max(1, int(sf1 * img.shape[1])), max(1, int(sf1 * img.shape[0]))), interpolation=random.choice([1, 2, 3]))
    return np.clip(img, 0.0, 1.0)


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:  # add color Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:  # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.0
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0, 0, 0], np.abs(L**2 * conv), img.shape[:2]).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img * np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img * np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.0
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img * np.random.multivariate_normal([0, 0, 0], np.abs(L**2 * conv), img.shape[:2]).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.0
    vals = 10**(2 * random.random() + 2.0)
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.0
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    return np.clip(img, 0.0, 1.0)


def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    return cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)


def degradation_bsrgan(img, sf=4, lq_patchsize=72):
    """BSRGAN practical degradation model.

    Args:
        img: HxWxC, [0, 1] float32
        sf: scale factor
        lq_patchsize: low quality patch size

    Returns:
        img: low-quality patch, size: lq_patchsize x lq_patchsize x C, range: [0, 1]
        hq: corresponding high-quality patch, size: (lq_patchsize*sf) x (lq_patchsize*sf) x C, range: [0, 1]
    """
    jpeg_prob, scale2_prob = 0.9, 0.25
    sf_ori = sf

    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]
    h, w = img.shape[:2]

    # If image is smaller than patch size, resize first
    target_hq_size = lq_patchsize * sf
    if h < target_hq_size or w < target_hq_size:
        scale_pad = max(target_hq_size / h, target_hq_size / w) * 1.05
        img = cv2.resize(img, (int(w * scale_pad), int(h * scale_pad)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    hq = img.copy()

    if sf == 4 and random.random() < scale2_prob:
        if np.random.rand() < 0.5:
            img = cv2.resize(img, (max(1, int(1 / 2 * img.shape[1])), max(1, int(1 / 2 * img.shape[0]))), interpolation=random.choice([1, 2, 3]))
        else:
            img = imresize_np(img, 1 / 2, True)
        img = np.clip(img, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(6), 6)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if i == 0 or i == 1:
            img = add_blur(img, sf=sf)
        elif i == 2:
            a, b = img.shape[1], img.shape[0]
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                img = cv2.resize(img, (max(1, int(1 / sf1 * img.shape[1])), max(1, int(1 / sf1 * img.shape[0]))), interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial_gaussian(25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()
                img = ndimage.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]
            img = np.clip(img, 0.0, 1.0)
        elif i == 3:
            img = cv2.resize(img, (max(1, int(1 / sf * a)), max(1, int(1 / sf * b))), interpolation=random.choice([1, 2, 3]))
            img = np.clip(img, 0.0, 1.0)
        elif i == 4:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 5:
            if random.random() < jpeg_prob:
                img = add_JPEG_noise(img)

    img = add_JPEG_noise(img)

    # Random crop paired patches
    h_lq, w_lq = img.shape[:2]
    if h_lq < lq_patchsize or w_lq < lq_patchsize:
        img = cv2.resize(img, (lq_patchsize, lq_patchsize), interpolation=cv2.INTER_LINEAR)
        hq = cv2.resize(hq, (lq_patchsize * sf_ori, lq_patchsize * sf_ori), interpolation=cv2.INTER_CUBIC)
    else:
        rnd_h = random.randint(0, h_lq - lq_patchsize)
        rnd_w = random.randint(0, w_lq - lq_patchsize)
        img = img[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

        rnd_h_H, rnd_w_H = int(rnd_h * sf_ori), int(rnd_w * sf_ori)
        hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf_ori, rnd_w_H:rnd_w_H + lq_patchsize * sf_ori, :]

    return img, hq
