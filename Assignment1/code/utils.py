from __future__ import division
from __future__ import print_function

import cv2
from PIL import Image
import numpy as np

def load_image(path):
  # load an image 
  img = cv2.imread(path)
  img = img[:, :, ::-1]  # BGR -> RGB
  return img

def save_image(path, img):
  img = img.copy()[:,:,::-1]
  return cv2.imwrite(path, img) 

def resize_image(img, new_size, interpolation):
  # resize an image into new_size (w * h) using specified interpolation
  # opencv has a weird rounding issue & this is a hacky fix
  # ref: https://github.com/opencv/opencv/issues/9096
  mapping_dict = {cv2.INTER_NEAREST: Image.NEAREST}
  if interpolation in mapping_dict:
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(new_size, 
                             resample=mapping_dict[interpolation])
    img = np.array(pil_img)
  else:
    img = cv2.resize(img, new_size, 
                     interpolation=interpolation)
  return img