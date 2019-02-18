#assignment1
#deepandas
#ddas27

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import random
import numpy as np

import cv2
import numbers
import collections
from PIL import Image

from utils import resize_image


# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################

class Compose(object):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     Scale(320),
      >>>     RandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str

class RandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly 
     (with a probability of 0.5).
  """
  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      img = cv2.flip(img, 1)
      return img
    return img

  def __repr__(self):
    return "Random Horizontal Flip"

#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation. 
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LANCZOS|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_LINEAR
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int) 
            or (isinstance(size, collections.Iterable) 
                and len(size) == 2)
           )
    self.size = size
    # use bilinear if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be scaled.

    Returns:
        numpy array: Rescaled image
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    # scale the image
    if isinstance(self.size, int):
      #################################################################################
      # Fill in the code here
      height, width, depth = img.shape
      if (width<height):
        img = resize_image(img,(self.size,round(self.size*height/width)), interpolation)
      else:
        img = resize_image(img,(round(self.size*width/height),self.size), interpolation)

      #################################################################################
      return img
    else:
      #################################################################################
      # Fill in the code here
      rows= self.size[0]
      cols= self.size[1]
      img = resize_image(img,(cols,rows), interpolation)
      #################################################################################
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])

class RandomSizedCrop(object):
  """Crop the given numpy array to random area and aspect ratio.

  A crop of random area of the original size and a random aspect ratio 
  of the original aspect ratio is made. This crop is finally resized to given size.
  This is widely used as data augmentation for training image classification models

  Args:
      size (sequence or int): size of target image. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int, 
          output size will be (size, size).
      interpolations (list of int, optional): Desired interpolation. 
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LANCZOS|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      area_range (list of int): range of the areas to sample from
      ratio_range (list of int): range of aspect ratio to sample from
      num_trials (int): number of sampling trials
  """

  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS, 
               area_range=(0.25, 1.0), ratio_range=(0.8, 1.2), num_trials=10):
    self.size = size
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations
    self.num_trials = int(num_trials)
    self.area_range = area_range
    self.ratio_range = ratio_range

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    for attempt in range(self.num_trials):

      # sample target area / aspect ratio from area range and ratio range
      area = img.shape[0] * img.shape[1]
      target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
      aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

      #################################################################################
      # Fill in the code here
      h, w, d = img.shape
      crop_w = math.floor(math.sqrt(target_area * aspect_ratio)) + 1
      crop_h = math.floor(math.sqrt(target_area / aspect_ratio)) + 1

      #Considering crop_w/crop_h = aspect ratio
      if(crop_w <=w and crop_h<=h):
        point_h = round(random.uniform((crop_h//2+1), (h - crop_h//2)))
        point_w = round(random.uniform((crop_w//2+1), (w - crop_w//2)))
        img = img[(point_w-(crop_w//2)):(point_w+(crop_w//2)), (point_h-(crop_h//2)):(point_h+(crop_h//2))]

      #Considering crop_h/crop_w = aspect ratio
      elif(crop_w <= h and crop_h <=w):
        point_w = round(random.uniform((crop_h // 2 + 1), (w - crop_h // 2)))
        point_h = round(random.uniform((crop_w // 2 + 1), (h - crop_w // 2)))
        img = img[(point_w - (crop_h // 2)):(point_w+(crop_h // 2)), (point_h - (crop_w // 2)): (point_h + (crop_w // 2))]

      #Case where (crop_w, crop_h) is not a feasible combination in either alignment
      else:
        continue

      if(isinstance(self.size, int)):
        img = resize_image(img, (self.size, self.size), interpolation)
        return img
      else:
        rows = self.size[0]
        cols = self.size[1]
        img = resize_image(img, (cols,rows), interpolation)
        return img
      #################################################################################
      # compute the width and height
      # note that there are two possibilities
      # crop the image and resize to output size

    # Fall back
    if isinstance(self.size, int):
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      #################################################################################
      # Fill in the code here
      height, width, depth = img.shape
      if(width<height):
        img = img[round((height-width)*0.5):round((height+width)*0.5),:]
      else:
        img = img[:,round((height - width) * 0.5):round((height + width) * 0.5)]
      img = resize_image(img, (self.size, self.size), interpolation)
      #################################################################################
      # with a square sized output, the default is to crop the patch in the center 
      # (after all trials fail)
      return img
    else:
      # with a pre-specified output size, the default crop is the image itself
      img = resize_image(img, (self.size, self.size), interpolation)
      '''
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      '''
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Random Crop" + \
           "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}%; Ratio {:.2f} - {:.2f}%]".format(
            target_size[0], target_size[1], 
            self.area_range[0], self.area_range[1],
            self.ratio_range[0], self.ratio_range[1])


class RandomColor(object):
  """Perturb color channels of a given image
  Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel. 
  The sampling is done independently for each channel.

  Args:
      color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
  """
  def __init__(self, color_range):
    self.color_range = color_range

  def __call__(self, img):
    #################################################################################
    # Fill in the code here
    alpha_red = random.uniform(-self.color_range, self.color_range)
    alpha_green = random.uniform(-self.color_range, self.color_range)
    alpha_blue = random.uniform(-self.color_range, self.color_range)

    img1 = np.array(img)
    img1[:,:,0] = img[:,:,0]*(1.+alpha_red)
    img1[:,:,1] = img[:,:,1]*(1.+alpha_green)
    img1[:,:,2] = img[:,:,2]*(1.+alpha_blue)
    #Clipping values to 255
    img1 = np.clip(img1, a_min=0, a_max=255)
    #################################################################################
    return img1

  def __repr__(self):
    return "Random Color [Range {:.2f} - {:.2f}%]".format(
            1-self.color_range, 1+self.color_range)


class RandomRotate(object):
  """Rotate the given numpy array (around the image center) by a random degree.

  Args:
      degree_range (float): range of degree (-d ~ +d)
  """
  def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
    self.degree_range = degree_range
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample rotation
    degree = random.uniform(-self.degree_range, self.degree_range)
    # ignore small rotations
    if np.abs(degree) <= 1.0:
      return img

    #################################################################################
    # Fill in the code here
    img = Image.fromarray(np.uint8(img))
    width = img.size[0]
    height = img.size[1]

    img = img.rotate(degree, expand = True, resample = Image.BICUBIC)

    W = img.size[0] #X
    H = img.size[1] #Y

    angle_a = math.radians(abs(degree))
    angle_b = math.radians(90 - degree)

    angle_a_sin = math.sin(angle_a)
    angle_b_sin = math.sin(angle_b)

    E = (math.sin(angle_a))/(math.sin(angle_b))* (H-W*(math.sin(angle_a)/math.sin(angle_b)))
    E = E/ 1 - (math.sin(angle_a)**2/math.sin(angle_b)**2)
    B = W-E
    A = (math.sin(angle_a)/math.sin(angle_b))*B

    img = img.crop((int(round(E)),int(round(A)), int(round(W-E)), int(round(H-A))))

    img = np.asarray(img)

    img = resize_image(img, (width,height), interpolation)

    #################################################################################
    # get the max rectangular within the rotated image
    return img

  def __repr__(self):
    return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range)