# coding: utf-8
# =====================================================================
#  Filename:    blur_util.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Utility functions for the blur detection script(s)
#
#  Note: Requires opencv 3.4.2 or later
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
import cv2


def blur_detector(image, size=60, threshold=10, draw=False):
    """
     Detects blur in images
    :param image: Image to detect blur in
    :param size: zero the FFT shift for this sized radius
    :param threshold: threshold at which image is considered blurred
    :param draw: flag to visualize FFT
    :return: FFT mean along with a blur boolean
    """
    height, width = image.shape
    center_x, center_y = int(height / 2), int(width / 2)

    # Perform FFT and shift to centre
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    if draw:
        # visualize FFT
        mag = 20 * np.log(np.abs(fft_shift))

        fig, ax = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(mag, cmap='gray')
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.show()

    # zero out FFT around center
    fft_shift[center_y - size: center_y + size, center_x - size: center_x + size] = 0

    # inverse FFT shift
    fft_shift = np.fft.ifftshift(fft_shift)
    inv_fft = np.fft.ifft2(fft_shift)

    # calculate mean magnitude of the spectrum
    mag = 20 * np.log(np.abs(inv_fft))
    mean = np.mean(mag)

    # image is blurred if the mean is less than the threshold
    return mean, mean <= threshold


def test_detection(image_in, threshold, vis):
    """
    Progressively blur the image to test the point at which image is considered blurred
    :param image_in: image to test
    :param threshold: threshold at which image is considered blurred
    :param vis: visualize the results flag
    """
    # iterate over progressively increasing blur radius
    for radius in range(1, 30, 2):
        image = image_in.copy()

        if radius > 0:

            # apply blur
            image = cv2.GaussianBlur(image, (radius, radius), 0)

            # check blur
            mean, blur = blur_detector(image, threshold=threshold, draw=vis > 0)
            image = np.dstack([image] * 3)

            color = (0, 0, 255) if blur else (0, 255, 0)
            info = f'Blurry ({mean})' if blur else f'Not Blurry ({mean})'
            cv2.putText(image, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            print(f'[INFO] Kernel: {radius}, Result: {info}')

            cv2.imshow("Blur Testing", image)
            cv2.waitKey(0)