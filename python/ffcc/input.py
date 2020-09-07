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
"""Input module for TF.Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import tensorflow.compat.v1 as tf


def input_batch_fn(filenames, batch_size, num_epochs, shuffle=True):
  """Construct an input pipeline that feeds into the TF graph.

  Produce input in batches given the list of filenames and number of epochs.

  Args:
    filenames: list(string), the list of TFRecord filenames of the training
      data.
    batch_size: int, the size of the batch for training.
    num_epochs: int, the number of epochs for the training.
    shuffle: (Optional) boolean, to shuffle the inputs. Default is True.

  Returns:
    A tuple of:
      features: a dictionary single example of feature columns.
        The dictionary with the following keys:
          burst_id: an unique id in the format of
            '[A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9]_YYYYDDMM_[0-9][0-9][0-9]'.
          rgb: the RGB image
          extended_feature: the value of the extended feature.
      labels: a dictionary label of a single example
        The dictionary with the following keys:
          illuminant: the color of illuminant in RGB (an unit vector). This is
            the reciprocal of RGB gains.
  """

  def _parse_function(example_proto):
    """Parse the features from the TFRecords."""
    features = {
        'name': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64),
        'rgb': tf.FixedLenFeature([], tf.string),
        'extended_feature': tf.FixedLenFeature([], tf.float32),
        'weight': tf.FixedLenFeature([], tf.float32),
        'illuminant': tf.FixedLenFeature([3], tf.float32)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    width = parsed_features['width']
    height = parsed_features['height']
    channel = parsed_features['channel']

    rgb = tf.decode_raw(parsed_features['rgb'], tf.float32)
    rgb = tf.reshape(rgb, [height, width, channel])

    name = parsed_features['name']
    extended_feature = parsed_features['extended_feature']
    illuminant = parsed_features['illuminant']
    weight = parsed_features['weight']

    labels = dict({'illuminant': illuminant})
    features = dict({
        'name': name,
        'rgb': rgb,
        'extended_feature': extended_feature,
        'weight': weight
    })
    return features, labels

  # Since dataset.shuffle is not randomly sampled over the whole dataset
  # (limited by the buffer size), we randomize the file names instead.
  if shuffle:
    # Fixed the random seed.
    random.Random(0).shuffle(filenames)

  num_parallel = 128
  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel)
  dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel)

  if shuffle:
    # Although we already (optionally) shuffle filenames, this shuffle gives us
    # batches that are random across epochs.
    dataset = dataset.shuffle(buffer_size=1024, seed=0)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(1)

  return dataset


def input_builder_stratified(data_dir,
                             batch_size,
                             num_epochs,
                             bucket_size,
                             train_set=None,
                             eval_set=None):
  """Stratifies the dataset into training and evaluation set for TF.Estimator.

  Args:
    data_dir: string, the dataset directory that contains *.tfrecord.
    batch_size: int, the batch size for the training set.
    num_epochs: int, the number of epochs for the training.
    bucket_size: int, the size of the bucket for stratification. The entire
      dataset is divided into buckets. The first element inside each bucket will
      be chosen as evaluation set, while the rest will be put into the training
      set. For example, if the bucket_size=2, then training and evaluation set
      will have equal size.
    train_set: list(string). Override the training set if the list is provided.
      The list contains the basename of each *.tfrecord for training.
    eval_set: list(string). Override the evaluation set if the list is provided.
      The list contains the basename of each *.tfrecord for training. Both
      train_set and eval_set have to be provided in order to take effect. If
      either eval_set or train_set are None, then the final training set will
      also include the eval set.

  Returns:
    The tuple of training, evalution set for TF.Estimator, final training set,
    final eval set, and total training iteration.
  """

  # List all the *.tfrecords from the data_dir and sort by burst_id
  filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, '*.tfrecord')))

  # Stratified the filenames to k buckets
  train_tfrecords = []
  eval_tfrecords = []
  if train_set and eval_set:
    # Check if training set and eval set overlapps.
    common = list(set(train_set) & set(eval_set))
    if common:
      raise ValueError('Training and eval set are not mutually exclusive. '
                       'Found common set: {}'.format(common))

    records = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    missing = list(set(records).difference(set(train_set) | set(eval_set)))
    if missing:
      raise ValueError('Records exists that are in neither the training set '
                       'nor the eval set: {}.'.format(missing))

    tf.logging.info('Creating training set and eval set from the list.')
    for f in filenames:
      basename = os.path.splitext(os.path.basename(f))[0]
      if basename in train_set:
        train_tfrecords.append(f)
      if basename in eval_set:
        eval_tfrecords.append(f)
  else:
    tf.logging.info(
        'Creating training and eval set from the stratified sampling.')
    for i, f in enumerate(filenames):
      if i % bucket_size == 0:
        eval_tfrecords.append(f)
      else:
        train_tfrecords.append(f)
    # No eval set is specified. We will train everything.
    train_tfrecords.extend(eval_tfrecords)
    random.shuffle(train_tfrecords)

  tf.logging.info('Training set size=%s. Eval set size=%s',
                  len(train_tfrecords), len(eval_tfrecords))

  def train_input_fn():
    return input_batch_fn(
        train_tfrecords, batch_size=batch_size, num_epochs=num_epochs)

  def eval_input_fn():
    return input_batch_fn(
        eval_tfrecords, batch_size=batch_size, num_epochs=1, shuffle=False)

  dataset_size = num_epochs * len(train_tfrecords)
  total_training_iterations = int(
      math.ceil(float(dataset_size) / float(batch_size)))

  return (train_input_fn, eval_input_fn, train_tfrecords, eval_tfrecords,
          total_training_iterations)
