# -*- coding: utf-8 -*-

import imageio
import numpy as np
import scipy.ndimage

#save image function
def save_img(im, ext=''):
    return imageio.imsave(f'{ext}.jpg', im=im)

#dodge function
def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')

#gray scale function
def grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

#loading image function
    
def load_image(img):
    start_img = imageio.imread(img)
    return start_img
#invert
def invert(im):
    return 255 - im

#bluring
def blur(im, s=20):
    return scipy.ndimage.filters.gaussian_filter(im, sigma=s)

#main function
def main(image, final_name='final', sigma=20, save=True):
    start_img = load_image(image)
    gray_img = grayscale(start_img)
    inverted_img = invert(gray_img)
    blur_img = blur(inverted_img, sigma)
    final_img = dodge(blur_img, gray_img)
    if save:    
        save_img(final_img, str(final_name))
    else:
        pass
    return final_img
#Loading Image
img = "Me&Yulya.jpg"

final_img = main(img, f'{img}_sketch', sigma=10, save=True)
save_img(final_img, 'final')

