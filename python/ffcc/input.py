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
import tensorflow as tf


def input_batch_fn(input_set, labels, batch_size, num_epochs, shuffle=True):
  """Construct an input pipeline that feeds into the TF graph.

  Produce input in batches given the list of data dictionary and number of
    epochs.

  Args:
    input_set: list(dictionary), the list of data dictionary with the following
      keys:
        name: full-path for the image file
        rgb: rgb image (float32)
        extended_feature: extended feature value for this sample (float32)
        weight: weight of this sample set (float32)
    labels: list(dictionary), the list of data dictionary with the following
      keys:
        illuminant: color of the illuminant in RGB (float32)
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

  dataset = tf.data.Dataset.from_tensor_slices((input_set, labels))
  if shuffle:
    # Although we already (optionally) shuffle filenames, this shuffle gives us
    # batches that are random across epochs.
    dataset = dataset.shuffle(buffer_size=1024, seed=0)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(1)

  return dataset


def input_builder_stratified(input_set, labels, batch_size, num_epochs):
  """Stratifies the dataset into training and evaluation set for TF.Estimator.

  Args:
    input_set: list(dictionary), each element has the following keys:
      name: full-path for the image file
      rgb: rgb image (float32)
      extended_feature: extended feature value for this sample (float32)
      weight: weight of this sample set (float32)
    labels: list(dictionary), each element has the following key:
        illuminant: color of the illuminant in RGB (float32)
    batch_size: int, the batch size for the training set.
    num_epochs: int, the number of epochs for the training.


  Returns:
    The tuple of training, evalution set for TF.Estimator, final training set,
    final eval set, and total training iteration.
  """

  tf.compat.v1.logging.info('Creating training set and eval set from the list.')

  def input_fn():
    return input_batch_fn(input_set, labels, batch_size=batch_size,
                          num_epochs=num_epochs)

  dataset_size = num_epochs * len(input_set)
  total_training_iterations = int(
      math.ceil(float(dataset_size) / float(batch_size)))

  return (input_fn, total_training_iterations)
