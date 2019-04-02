from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
from torch.utils import data
from utils import load_image
import numpy as np


class MiniPlacesLoader(data.Dataset):
  """
  A simple dataloader for mini places
  """
  def __init__(self,
               root_folder,
               label_file=None,
               num_classes=100,
               split="train",
               transforms=None):
    assert split in ["train", "val", "test"]
    # root folder, split
    self.root_folder = root_folder
    self.split = split
    self.transforms = transforms
    self.n_classes = num_classes

    # load all labels
    if label_file is None:
      label_file = os.path.join(root_folder, split + '.txt')
    if not os.path.exists(label_file):
      raise ValueError(
        'Label file {:s} does not exist!'.format(label_file))
    with open(label_file) as f:
      lines = f.readlines()

    # store the file list
    file_label_list = []
    for line in lines:
      filename, label_id = line.rstrip('\n').split(' ')
      label_id = int(label_id)
      filename = os.path.join(root_folder, "images", filename)
      file_label_list.append((filename, label_id))

    self.file_label_list = file_label_list

  def __len__(self):
    return len(self.file_label_list)

  def __getitem__(self, index):
    # load img and label
    filename, label_id = self.file_label_list[index]
    img = np.ascontiguousarray(load_image(filename))
    label = label_id

    # apply data augmentation
    if self.transforms is not None:
      img  = self.transforms(img)
    return img, label

  def get_index_mapping(self):
    # load the train label file
    train_label_file = os.path.join(self.root_folder, self.split + '.txt')
    if not os.path.exists(train_label_file):
      raise ValueError(
        'Label file {:s} does not exist!'.format(label_file))
    with open(train_label_file) as f:
      lines = f.readlines()

    # get the category names
    id_index_map = {}
    for line in lines:
      filename, label_id = line.rstrip('\n').split(' ')
      cat_name = filename.split('/')[-2]
      id_index_map[label_id] = cat_name

    return id_index_map
