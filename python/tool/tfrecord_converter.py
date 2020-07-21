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
r"""Encoding the data to TFRecord Format.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from . import io
import tensorflow.compat.v1 as tf


tf.app.flags.DEFINE_string('input_dir', '',
                           'Input dir for the training data.')
tf.app.flags.DEFINE_string('output_dir', '',
                           'Output dir for the encoded training data.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if not os.path.isdir(FLAGS.input_dir):
    tf.logging.error('Invalid input directory: {}'.format(FLAGS.input_dir))
    return

  if not os.path.isdir(FLAGS.output_dir):
    tf.logging.error('Invalid output directory: {}'.format(FLAGS.output_dir))
    return

  tf.logging.info('Loading dataset')
  dataset = io.read_dataset_from_files(FLAGS.input_dir)

  tf.logging.info('Encoding dataset')
  io.write_dataset_to_tfrecord(dataset, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run(main)
