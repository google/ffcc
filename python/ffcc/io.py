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
from skimage import io as sio
import tensorflow.compat.v1 as tf


def read_from_directory(dir_path):
  r"""Reads data from the directory.

  Args:
    dir_path: directory path that contains the input data, The structure of the
      directory.
        dir_path/
          0/                                    # 0 .. n-1 frame
            image.tiff                          # RGB stats in float32
            extended_feature.txt                # extended feature value.
          illuminant.txt                        # color of the illuminant in RGB
          weight.txt                            # weighting

  Returns:
    rgb_illuminant: color of the illuminant in RGB (float32).
    weight: weight of this training set (float32).
    frames: a list of tuple of:
      rgb: RGB image (float32)
      extended_feature: value of the extended feature for this sample (float32)
  """
  # pyformat: enable

  # Make sure the illuminant is unit vector.
  rgb_illuminant = np.loadtxt(
      os.path.join(dir_path, 'illuminant.txt'), dtype=np.float32)
  rgb_illuminant /= np.linalg.norm(rgb_illuminant)
  weight = np.loadtxt(os.path.join(dir_path, 'weight.txt'), dtype=np.float32)

  frames = []
  for item in tf.io.gfile.listdir(dir_path):
    sub_dir = os.path.join(dir_path, item)
    if tf.io.gfile.isdir(sub_dir) and item.isdigit():
      try:
        rgb = sio.imread(os.path.join(sub_dir, 'image.tiff'))
        extended_feature = np.loadtxt(
            os.path.join(sub_dir, 'extended_feature.txt'), dtype=np.float32)
        frames.append((rgb, extended_feature))
      except:  # pylint: disable=bare-except
        tf.logging.error('Unable to load %s', sub_dir)

  return rgb_illuminant, weight, frames


def encode_to_tfrecord(data, output_dir):
  """Encodes a data item to TFRecord format.

  Args:
    data: A dictionary maps to the FFCC data bundle.
    output_dir: The output folder. The record will be saved as <name>.tfrecord.
  """

  def _int64_feature(value):
    """Converts a list of int64 into TF Feature."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=np.atleast_1d(value)))

  def _float_feature(value):
    """Converts a list of fp32 into TF Feature."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=np.atleast_1d(value)))

  def _bytes_feature(value):
    """Converts a list of bytes into TF Feature."""
    # Note: tf.train.BytesList does not work well with np.asleast_1d. If input
    # value is all 0s, then tf.train.BytesList would truncate the result from
    # np.asleast_1d(value) as a emptry string.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  name = data['name']
  rgb = data['rgb']
  extended_feature = data['extended_feature']
  illuminant = data['illuminant']
  weight = data['weight']

  filename = os.path.join(output_dir, name + '.tfrecord')
  writer = tf.python_io.TFRecordWriter(filename)

  height, width, channel = rgb.shape
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'name': _bytes_feature(name.encode('utf-8')),
              'height': _int64_feature(height),
              'width': _int64_feature(width),
              'channel': _int64_feature(channel),
              'rgb': _bytes_feature(rgb.tobytes()),
              'extended_feature': _float_feature(extended_feature),
              'illuminant': _float_feature(illuminant),
              'weight': _float_feature(weight)
          }))
  writer.write(example.SerializeToString())
  writer.close()


def decode_from_tfrecord_proto(example_proto):
  """Decodes TFRecord protobuf.

  Args:
    example_proto: TF example protobuf.

  Returns:
    A single FFCC training feature.
  """
  features = {
      'name': tf.FixedLenFeature([], tf.string),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'channel': tf.FixedLenFeature([], tf.int64),
      'rgb': tf.FixedLenFeature([], tf.string),
      'extended_feature': tf.FixedLenFeature([], tf.float32),
      'illuminant': tf.FixedLenFeature([3], tf.float32),
      'weight': tf.FixedLenFeature([], tf.float32)
  }

  parsed_features = tf.parse_single_example(example_proto, features)
  width = parsed_features['width']
  height = parsed_features['height']
  channel = parsed_features['channel']

  rgb = tf.decode_raw(parsed_features['rgb'], tf.float32)
  rgb = tf.reshape(rgb, [height, width, channel])
  parsed_features['rgb'] = rgb
  return parsed_features


def read_dataset_from_files(path):
  """Read the training data from a list of folders contains raw data.

  Args:
    path: The root folder of the dataset.

  Returns:
    A list of dictionary represented the bundles of data.
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
  data = []
  for folder in folders:
    try:
      illuminant, weight, frames = read_from_directory(folder)
      for i, f in enumerate(frames):
        rgb, extended_feature = f
        data.append({
            'name': '{}_{}'.format(os.path.basename(folder), i),
            'rgb': rgb,
            'extended_feature': extended_feature,
            'illuminant': illuminant,
            'weight': weight,
        })
    except:  # pylint: disable=bare-except
      tf.logging.error('Unable to load %s', folder)

  return data


def write_dataset_to_tfrecord(dataset, output_dir):
  """Encodes dataset into TFRecords and write into the destination folder."""
  for d in dataset:
    encode_to_tfrecord(d, output_dir)
