# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FFCC I/O module that parses data from the filesystem."""

import os
import numpy as np
import tensorflow as tf
import random


def read_image_data(scene_name, dir_path):
  r"""Reads scene data from the directory.

  Args:
    scene_name: scene name (e.g., '000001')
    dir_path: directory path that contains the input data. If the
    scene_name was '000001', for example, then the directory should contain
    the following files:
            000001.png                        # Linear RGB image
            000001.txt                        # color of the illuminant in RGB
    ----------------------- optional files ---------------------------------
            000001_extended_feature.txt       # extended feature value
            000001_weight.txt                 # loss weighting

  Returns:
    rgb_illuminant: color of the illuminant in RGB (float32).
    weight: weight of this training set (float32). If not
      'scene_name_weight.txt' does not exist, it will be set to 1.
    rgb: RGB image (float32)
    extended_feature: value of the extended feature for this sample (
      float32). If not 'scene_name_extended_feature.txt' does not exist,
      it will be set to 1.
  """
  # load the weight if the weight file exists
  if tf.io.gfile.exists(os.path.join(dir_path, scene_name + '_weight.txt')):
    weight = np.loadtxt(os.path.join(dir_path, scene_name + '_weight.txt'),
                        dtype=np.float32)
  else:
    weight = np.float32(1.0)
  # load the extended feature if the weight file exists
  if tf.io.gfile.exists(os.path.join(dir_path,
                                     scene_name + '_extended_feature.txt')):
    extended_feature = np.loadtxt(os.path.join(
      dir_path, scene_name + '_extended_feature.txt'), dtype=np.float32)
  else:
    extended_feature = np.float32(1.0)
  # load image and illuminant data
  rgb = tf.io.decode_image(tf.io.read_file(
    os.path.join(dir_path, scene_name + '.png')), dtype=tf.dtypes.uint16)
  normalized = tf.cast(rgb, dtype=tf.float32) / tf.uint16.max
  tf.compat.v1.disable_eager_execution()
  rgb = normalized.eval(session=tf.compat.v1.Session())
  # Make sure the illuminant is unit vector.
  rgb_illuminant = np.loadtxt(
    os.path.join(dir_path, scene_name + '.txt'), dtype=np.float32)
  rgb_illuminant /= np.linalg.norm(rgb_illuminant)
  return rgb_illuminant, weight, rgb, extended_feature


def get_training_eval_sets(files, cvfolds, test_fold):
  """Retrieve train and evaluation sets for the current testing fold .

  Args:
    files: A list of filenames that should be synched with the cvfolds.
    cvfolds: A list of corresponding fold number of each filename in 'files'
    test_fold: Number of testing fold (should be in the range [1-3]).

  Returns:
    train_set and eval_set, lists of training and evaluation
      filenames, respectively.
  """
  assert len(files) == len(cvfolds)
  assert test_fold in range(1,4)
  train_folds = list({1, 2, 3} - {test_fold})
  train_set = []
  for train_fold_i in train_folds:
    train_set = np.append(train_set,
                          np.array(files).transpose()[
                            (cvfolds == train_fold_i)])
  train_set = train_set.tolist()

  eval_set = np.array(files)[(cvfolds == test_fold)]
  eval_set = eval_set.tolist()
  return train_set, eval_set


def build_dataset_dict(files, shuffle):
  """Encode training data into input/label dictionaries.

  Args:
    files: full path for scene names
    shuffle: (Optional) boolean, to shuffle the inputs. Default is True.
  Returns:
    input: a dictionary with the following keys:
        name: full-path for image files
        rgb: rgb images (float32)
        extended_feature: extended feature values (float32)
        weight: weights (float32)

    label: a dictionary with the following key:
        illuminant: illuminant RGB colors (float32)
  """

  illuminant_set = []
  weight_set = []
  rgb_set = []
  extended_feature_set = []
  for f in iter(files):
    scene = os.path.splitext(os.path.basename(f))[0]
    scene_path = os.path.split(f)[0]
    illuminant, weight, rgb, extended_feature = read_image_data(
      scene, scene_path)
    illuminant_set.append(illuminant)
    weight_set.append(weight)
    rgb_set.append(rgb)
    extended_feature_set.append(extended_feature)

  if shuffle:
    # Fixed the random seed.
    data = list(zip(files, illuminant_set, weight_set,
                    rgb_set, extended_feature_set))
    random.Random(0).shuffle(data)
    (files, illuminant_set, weight_set, rgb_set,
     extended_feature_set) = zip(*data)

  input = {'name': np.array(files),
           'rgb': np.array(rgb_set),
           'extended_feature': np.array(extended_feature_set),
           'weight': np.array(weight_set)}
  label = {'illuminant': np.array(illuminant_set)}

  return input, label


def read_dataset_from_dir(path, test_fold, shuffle=True):
  """Read the training data from the given directory and sub-directory(s).

  Args:
    path: The root directory of the dataset.
    test_fold: test fold number; should be in the range [1-3].
    shuffle: (Optional) boolean, to shuffle the inputs. Default is True.
  Returns:
    training_input, training_label, eval_input, and eval_label.
    Both training_input and eval_input are training and evaluation
    data dictionaries, respectively. Each training/evaluation data dictionary
    has the following keys:
        name: full-path for the image files
        rgb: rgb images (float32)
        extended_feature: extended feature values (float32)
        weight: weights (float32)
      Both train_labels and eval_labels have the following key:
        illuminant: illuminant RGB colors (float32)
  """

  def _list_subfolders(path):
    """List the immediate subfolders beneath the path.

    Args:
      path: the path to be scanned.

    Returns:
      A list of subfolders.
    """
    subfolders = []
    for f in tf.io.gfile.listdir(path):
      full_path = os.path.join(path, f)
      if tf.io.gfile.isdir(full_path):
        subfolders.append(f)
    return subfolders

  folders = [os.path.join(path, f) for f in _list_subfolders(path)]
  # include the main directory
  folders.append(path)
  image_files = []
  for folder in folders:
    if folder:
      image_files = [
        os.path.join(folder, file) for file in os.listdir(folder) if
        file.endswith('.png')]

  # check if cvfolds.txt file exists in the root path
  if tf.io.gfile.exists(os.path.join(path, 'cvfolds.txt')):
    cvfolds = np.loadtxt(os.path.join(path, 'cvfolds.txt'), dtype=np.int)
  else:
    # create a 3-fold cross-validation partition and store it in the root path
    number_of_files = len(image_files)
    indices = np.linspace(0, number_of_files - 1, number_of_files).astype(int)
    cvfolds = np.zeros(number_of_files).astype(int)
    random.shuffle(indices)
    for fold in range(3):
      cvfolds[indices[np.floor((fold * number_of_files) / 3).astype(int):
                      np.floor(((fold + 1) * number_of_files) / 3).astype(int)
              ]] = int(fold + 1)
    cvfolds_file = open(os.path.join(path, 'cvfolds.txt'), 'w')
    np.savetxt(cvfolds_file, cvfolds, fmt='%d')
    cvfolds_file.seek(cvfolds_file.tell() - 1, os.SEEK_SET)
    cvfolds_file.truncate()
    cvfolds_file.close()

  training_files, eval_files = get_training_eval_sets(
    image_files, cvfolds, test_fold)

  training_input, training_label = build_dataset_dict(training_files, shuffle)

  eval_input, eval_label = build_dataset_dict(eval_files, shuffle)

  return training_input, training_label, eval_input, eval_label