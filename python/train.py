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
"""FFCC Training and Evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from . import model
from . import ops
import input as ffcc_input
import tensorflow.compat.v1 as tf

# This directory stores the TFRecord for training.
DATA_DIR = ''

# Model dir for checkpoints, etc.
MODEL_DIR = ''

# The number of epochs for training.
NUM_EPOCHS = 150

# How often should checkpoints and summaries be saved and computed, for each
# training epoch. Higher numbers will result in more frequent tensorboard
# updates, at the expense of training efficiency.
NUM_UPDATES_PER_EPOCH = 10


def train_and_eval_fn(params, hparams, train_set, eval_set):
  """Creates A tf.estimator and the corresponding training and eval specs.

  Args:
    params: dictionary of model parameters.
    hparams: dictionary of hyperparameters.
    train_set: Name of the tfrecords to load as part of the train set.
    eval_set: Name of the tfrecords to load as part of the eval set.

  Returns:
    A tuple (estimator, train_spec, eval_spec).
  """

  tf.logging.info('hparams = %s', hparams)
  tf.logging.info('num_epochs = %s', NUM_EPOCHS)

  if train_set is not None and eval_set is not None:
    # Check that that training set and eval set are mutually exclusive.
    assert len(set(train_set) | set(eval_set)) == len(train_set + eval_set)
    # If the training set and eval set all have names with '_' in them, raise
    # a warning for each intersecting prefix.
    if all([s.rfind('_') != -1 for s in train_set]) and all(
        [s.rfind('_') != -1 for s in eval_set]):
      train_roots = [s[:s.rfind('_')] for s in train_set]
      eval_roots = [s[:s.rfind('_')] for s in eval_set]
      for s in set(train_roots) & set(eval_roots):
        print('WARNING: {}_* present in both training and eval sets'.format(s))

  (train_input_fn, eval_input_fn, _, _, total_training_iterations) = (
      ffcc_input.input_builder_stratified(DATA_DIR, hparams.batch_size,
                                          NUM_EPOCHS, hparams.bucket_size,
                                          train_set, eval_set))
  # Used when decaying values during training.
  hparams.add_hparam('total_training_iterations', total_training_iterations)

  update_steps = max(
      1, math.ceil(total_training_iterations / NUM_UPDATES_PER_EPOCH))
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=update_steps, save_summary_steps=update_steps)

  # Note: the max_steps need to be set otherwise the eval job on borg will be
  # stuck in infinite loop: b/130740041
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=total_training_iterations)

  # Just to be sure that evaluation is working, we manually force evaluation
  # for all batches in the eval set. Thresholds are set to small values so that
  # evaluation happens for all checkpoints as per `save_checkpoints_steps`.
  eval_steps = 10

  # Training the FFCC model is really fast; we set delays to 0s in order to
  # make intermediate evaluation results visible in tensorboard.
  eval_spec = tf.estimator.EvalSpec(
      name='default',
      input_fn=eval_input_fn,
      steps=eval_steps,
      start_delay_secs=0,
      throttle_secs=0)

  # Create the tf.Estimator instance
  estimator = tf.estimator.Estimator(
      model_fn=model.model_builder(hparams),
      params=params,
      config=run_config)
  return estimator, train_spec, eval_spec


def print_eval(params, hparams):
  """Printing the eval results on the latest checkpoint.

  This function will run on the entired TFRecord dataset and it is useful for
  validating the training result.

  Args:
    params: model params
    hparams: a tf.HParams object created by create_default_hparams().
  """
  (_, eval_input_fn, _, _, _) = \
      ffcc_input.input_builder_stratified(DATA_DIR,
                                          batch_size=1,
                                          num_epochs=1,
                                          bucket_size=1)

  run_config = tf.estimator.RunConfig()
  estimator = tf.estimator.Estimator(
      model_fn=model.model_builder(hparams),
      params=params,
      model_dir=MODEL_DIR,
      config=run_config)

  for result in estimator.predict(input_fn=eval_input_fn):
    burst_id = result['burst_id']
    uv = result['uv']
    # Convert UV to RGB illuminants for visualization
    with tf.Graph().as_default():
      with tf.Session() as sess:
        uv_batch = uv.reshape((1, 2))
        rgb = sess.run(ops.uv_to_rgb(uv_batch))[0]
    tf.logging.info('id=%s uv=%s rgb=%s', burst_id.decode('utf-8'), uv, rgb)


def main(_):
  # Something about this code causes optimization to perform very poorly when
  # run on a GPU, so as a safeguard we prevent the user from using CUDA.
  # TODO(barron/yuntatsai): Track down the source of this discrepancy (maybe
  # something involving the FFT gradients?).
  assert not tf.test.is_built_with_cuda()

  # TODO(fbleibel): populate params and hparams based on project files.
  hparams = {}
  params = {}

  # TODO(fbleibel): load train and eval set from disk.
  train_set = []
  eval_set = []

  estimator, train_spec, eval_spec = train_and_eval_fn(params, hparams,
                                                       train_set, eval_set)

  do_print_eval = False
  if do_print_eval:
    print_eval(params, hparams)
  else:
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
