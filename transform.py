#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:07:18 2020

@author: piotrek
"""

import numpy as np
import torch
from skimage import transform, color


class Rescale(object):

    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, genre = sample['image'], sample['genre']

        h, w = image.shape[:2]

        # only one dim provided
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'genre': genre}


class RandomCrop(object):
    """Crop randomly the image in a sample.

       Args:
           output_size (tuple or int): Output size. If int, square crop
               is made.
       """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, genre = sample['image'], sample['genre']

        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'genre': genre}


class ToTensor(object):
    """Convert ndarrays in sample to Tensor."""

    def __call__(self, sample):
        image, genre = sample['image'], sample['genre']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'genre': torch.from_numpy(genre)}


class ToRGB(object):
    """ Convert image to 3-channel RGB from RGBA """

    def __call__(self, sample):
        image, genre = sample['image'], sample['genre']
        image = color.rgba2rgb(image)

        return {'image': image, 'genre': genre}


class MeanSubstraction(object):
    """ Normalize by substracting mean value for each channel """
    """" image input format like H x W x C """

    def __call__(self, sample):
        image, genre = sample['image'], sample['genre']

        transformed = np.zeros(image.shape)

        # no. of channels as last value
        for d in range(image.shape[-1]):
            channel = image[:, :, d]
            mean = np.mean(channel)
            transformed[:, :, d] = channel - mean

        return {'image': transformed, 'genre': genre}


class MelspectrogramNormalize(object):
    """ Normalize by substracting mean value for each channel and divide by channels' std"""
    """" image input format like H x W x C """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, genre = sample['image'], sample['genre']

        transformed = np.zeros(image.shape)

        # no. of channels as last value
        for d in range(image.shape[-1]):
            mean_channel = self.mean[d]
            std_channel = self.std[d]
            channel = image[:, :, d]
            transformed[:, :, d] = (channel - mean_channel) / std_channel

        return {'image': transformed, 'genre': genre}


class ListToTensor(object):
    """Convert list in sample to Tensor."""

    def __call__(self, sample):
        mfcc, genre = sample['mfcc'], sample['genre']

        max_length = 650

        mfcc_tensor = torch.FloatTensor(mfcc).T

        shape = (mfcc_tensor.shape[0], max_length - mfcc_tensor.shape[1])
        zeros = torch.zeros(shape)

        mfcc_tensor = torch.cat((mfcc_tensor, zeros), dim=1)

        # n_coefficietns x length after transpose
        return {'mfcc': mfcc_tensor.T,
                'genre': torch.from_numpy(genre)}


class MfccNormalize(object):
    """ Normalize by substracting mean value and divide by std for each coefficient """
    """" mfcc input format like n_coeff x length """

    def __call__(self, sample):
        mfcc, genre = sample['mfcc'], sample['genre']

        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)

        return {
            'mfcc': mfcc.sub(mean).div(std),
            'genre': genre
        }


class SubtractMfccNormalize(object):
    """ Normalize by substracting mean value for each coefficient """
    """" mfcc input format like n_coeff x length """

    def __call__(self, sample):
        mfcc, genre = sample['mfcc'], sample['genre']

        mean = mfcc.mean(dim=1, keepdim=True)

        return {
            'mfcc': mfcc.sub(mean),
            'genre': genre
        }
