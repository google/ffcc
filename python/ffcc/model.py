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
"""The FFCC model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from ffcc import ellipse
from ffcc import fft
from ffcc import losses
from ffcc import ops
import numpy as np
import tensorflow as tf

# This number should match the number of histogram features (e.g. a 64x64xN
# histogram).
NUMBER_OF_CHANNELS_FILTERS = 2


def evaluate_model(rgb, extended_feature, filters_extended_fft,
                   filters_base_fft, bias_extended, bias_base, params):
  """Evaluate a FFCC model.

  More detail in Fast Fourier Color Constancy, Barron and Tsai, CVPR 2017:
  https://arxiv.org/abs/1611.07596

  Args:
    rgb: RGB images (float32) in the shape of [batch_size, height, width,
      channels].
    extended_feature: some feature particular to the input image in the shape of
      shape of [batch_size, extended_vector_length].
    filters_extended_fft: The FFT of stacked 2D filters in the shape of
      [extended_vector_length, height, width, channels], where the first rank of
      the tensor must have the same size of weights. The fused filter will be
      converted into 2D-DFT coefficients for each channel in the later stage of
      the TF graph. This will be learned.
    filters_base_fft: The FFT of stacked 2D filters in the shape of [height,
      width, channels]. This will be learned.
    bias_extended: 2D bias in the shape of [extended_vector_length, height,
      width], a TF tensor. This will be learned.
    bias_base: 2D bias in the shape of [height, width], a TF tensor. This will
      be learned.
    params: a dict with keys:
      'first_bin': (float) location of the edge of the first histogram bin.
      'bin_size': (float) size of each histogram bin.
      'nbins': (int) number of histogram bins.

  Returns:
    Tuple of:
      mu_adjusted: the center mass of the PMF in UV space, after being soft
        projected onto a ellipse defined in params, in the shape of
        [batch_size, 2].
      sigma: the covariance matrix of the PMF In UV space, in the shape of
        [batch_size, 2, 2].
      heatmap: the predicted heat map in UV space, with a size of [batch_size,
        n, n].
  """
  chroma_histograms, extended_features = ops.data_preprocess(
      rgb, extended_feature, params)

  def fuse(x, y):
    return tf.reduce_sum(
        x[tf.newaxis, :, :, :, :] * y[:, :, tf.newaxis, tf.newaxis, tf.newaxis],
        axis=1)

  filters_fft = tf.complex(
      fuse(tf.math.real(filters_extended_fft), extended_features),
      fuse(tf.math.imag(filters_extended_fft), extended_features)) + \
                filters_base_fft

  bias = tf.reduce_sum(
      bias_extended[tf.newaxis, :, :] *
      extended_features[:, :, tf.newaxis, tf.newaxis],
      axis=1) + bias_base

  with tf.name_scope('extended'):
    # Visualize the chroma histograms for a few datapoints.
    histograms_norm = tf.sqrt(
        chroma_histograms /
        (1e-7 + tf.reduce_max(chroma_histograms, axis=[1, 2], keepdims=True)))
    histograms_vis = tf.cast(
        tf.round(255. * tf.concat(tf.unstack(histograms_norm, axis=3), axis=2)),
        tf.uint8)[:, :, :, tf.newaxis]
    tf.summary.image('chroma_histograms', histograms_vis)

    # Visualize the filters and bias for a few datapoints.
    filters = fft.fftshift(ops.c2r_ifft2(filters_fft), axis=[1, 2])
    filters_quant = tf.cast(
        tf.round(
            127.5 *
            (filters /
             (1e-7 + tf.reduce_max(tf.abs(filters), axis=[1, 2], keepdims=True))
             + 1.)), tf.uint8)
    filters_vis = tf.concat(
        tf.unstack(filters_quant, axis=3), axis=2)[:, :, :, tf.newaxis]
    tf.summary.image('datapoint_filters', filters_vis)
    bias_min = tf.reduce_min(bias, axis=[1, 2], keepdims=True)
    bias_max = tf.reduce_max(bias, axis=[1, 2], keepdims=True)
    bias_vis = tf.cast(
        tf.round(255 * (bias - bias_min) / (bias_max - bias_min)), tf.uint8)
    tf.summary.image('datapoint_bias', bias_vis[:, :, :, tf.newaxis])

  heatmap = ops.eval_features(chroma_histograms, filters_fft, bias)

  mu_idx, sigma_idx = ops.bivariate_von_mises(ops.softmax2(heatmap))

  # Pad the sigma for better generalization.
  sigma_idx += params['variance'] * tf.eye(2)
  mu, sigma = ops.idx_to_uv(mu_idx, sigma_idx, params['bin_size'],
                            params['first_bin'])

  b_vec = np.asarray(params['ellipse_params']['b_vec'])
  w_mat = np.reshape(params['ellipse_params']['w_mat'], [len(b_vec)] * 2)
  mu_adjusted = ellipse.project(mu, w_mat, b_vec)

  with tf.name_scope('summaries'):
    ellipse_hinge = tf.math.maximum(0., ellipse.distance(mu, w_mat, b_vec) - 1.)
    tf.summary.scalar('sum_ellipse_hinge', tf.reduce_sum(ellipse_hinge))

  with tf.name_scope('outputs'):
    bins = (
        np.arange(params['nbins']) * params['bin_size'] + params['first_bin'])
    vv, uu = np.meshgrid(bins, bins)
    uuvv = np.stack([uu, vv], axis=-1)
    uvs = np.reshape(uuvv, [-1, 2])
    ellipse_vis = tf.reshape(
        ellipse.distance(uvs, w_mat, b_vec) >= 1., [params['nbins']] * 2)
    mu_vis = 0.
    n_lerps = 32
    for t in np.linspace(0, 1, n_lerps):
      mu_lerp = mu * t + mu_adjusted * (1. - t)
      mu_max_pmf = tf.reduce_max(
          ops.uv_to_pmf(mu_lerp, params['bin_size'], params['first_bin'],
                        params['nbins']),
          axis=0)
      col = np.array([t, 0, 1. - t])
      mu_vis = tf.math.maximum(mu_vis, mu_max_pmf[:, :, tf.newaxis] * col)
    mu_vis = tf.sqrt(mu_vis)
    mu_vis = tf.stack([
        mu_vis[:, :, 0], 0.25 * tf.cast(ellipse_vis, tf.float32), mu_vis[:, :,
                                                                         2]
    ],
                      axis=-1)
    tf.summary.image('mus_and_adjustment',
                     tf.cast(tf.round(255. * mu_vis[tf.newaxis]), tf.uint8))

  return mu_adjusted, sigma, heatmap


def latent_to_model(filters_extended_latent, filters_base_latent,
                    bias_extended_latent, bias_base_latent, hparams, params):
  """Converts the model from the latent (preconditioned) space to model space.

  The model is trained in preconditioned space, but the actual inference is
  evaluated in the model space. This function does all the conversion.

  Args:
    filters_extended_latent: The latent stacked 2D filters in the shape of
      [extended_vector_length, height, width, channels].
    filters_base_latent: The latent stacked 2D filters in the shape of [height,
      width, channels].
    bias_extended_latent: The latent 2D bias in the shape of
      [extended_vector_length, height, width].
    bias_base_latent: The latent 2D bias in the shape of [height, width].
    hparams: The hyperparameters.
    params: Model parameters.

  Returns:
    The tuple of (filters_extended_fft, filters_base_fft, bias_extended,
    bias_base, precond_filters, precond_bias).
      filters_extended_fft: The FFT of stacked 2D filters in the shape of
        [extended_vector_length, height, width, channels].
      filters_base_fft: The FFT of stacked 2D filters in the shape of [height,
        width, channels].
      bias_extended_fft: The FFT of 2D bias in the shape of
        [extended_vector_length, height, width].
      bias_base_fft: The FFT of 2D bias in the shape of [height, width].
      bias_extended: 2D bias in the shape of [extended_vector_length, height,
        width].
      bias_base: 2D bias in the shape of [height, width], a TF tensor.
      precond_filters: The preconditioner to the filters.
      precond_bias: The preconditioner to the bias.
  """

  n = params['nbins']
  precond_filters = tf.cast(
      fft.compute_preconditioner_vec(n, hparams['mult_filters_tv'],
                                     hparams['mult_filters_l2']), tf.float32)
  precond_bias = tf.cast(
      fft.compute_preconditioner_vec(n, hparams['mult_bias_tv'],
                                     hparams['mult_bias_l2']), tf.float32)
  filters_extended_fft = fft.vec_to_fft2(precond_filters[tf.newaxis, :, :] *
                                         filters_extended_latent)
  filters_base_fft = fft.vec_to_fft2(
      (precond_filters * filters_base_latent)[tf.newaxis, :, :])

  bias_extended_fft = fft.vec_to_fft2(
      (precond_bias[tf.newaxis, :, 0] * bias_extended_latent)[:, :, tf.newaxis])
  bias_base_fft = fft.vec_to_fft2(
      (precond_bias[:, 0] * bias_base_latent)[tf.newaxis, :, tf.newaxis])
  bias_extended = tf.squeeze(ops.c2r_ifft2(bias_extended_fft))
  bias_base = tf.squeeze(ops.c2r_ifft2(bias_base_fft))

  return (filters_extended_fft, filters_base_fft, bias_extended_fft,
          bias_base_fft, bias_extended, bias_base, precond_filters,
          precond_bias)


def model_builder(hparams):
  """Prepares custom model_fn for the tf.Estimator.

  Args:
    hparams: a dict with keys:
      'mult_bias_l2': float32 vector of shape [NUMBER_OF_CHANNELS_FILTERS]
      'mult_bias_tv': float32 vector of shape [NUMBER_OF_CHANNELS_FILTERS]
      'mult_smooth': see below.
      'total_training_iterations': total number of iterations (steps) over all
        training epochs.
      'learning_rate': a learning rate multiplier. See below.

  Returns:
    model_fn for tf.Estimator.
  """

  def _visualize_fft(filters_fft, shift=False):
    """Visualizes FFT filters.

    Args:
      filters_fft: 2D filters in Fourier coefficients in the shape of
        [batch_size, n, n, channels].
      shift: performs fftshift if True.

    Returns:
      A tensor to visualized filters in real domain in batch_size * channels
      grid, in the shape of [1, batch_size, channels, 1].
    """

    def _gen_vis(f, shift):
      f_shifted = fft.fftshift(f, axis=[0, 1]) if shift else f
      f_centered = f_shifted - tf.reduce_mean(f_shifted)
      return f_centered / tf.maximum(
          tf.reduce_max(tf.abs(f_centered)), sys.float_info.epsilon)

    filters_fft.shape.assert_has_rank(4)
    filters = ops.c2r_ifft2(filters_fft)
    batch_unpacked = tf.unstack(filters)
    rows = []
    for channels in batch_unpacked:
      rows.append(
          tf.concat([_gen_vis(f, shift) for f in tf.unstack(channels, axis=-1)],
                    axis=1))
    return tf.concat(rows, axis=0)[tf.newaxis, :, :, tf.newaxis]

  def _model_fn(features, labels, mode, params):
    """Model function for tf.Estimator.

    Args:
      features (dict): A dictionary maps to a batch of training features.
        The following keys are expected:
          {'name': an unique id for this example.
           'rgb': the RGB image.
           'extended_feature': a (float) number representing some unique feature
             of this example.
      labels (dict): A dictionary map to a batch of ground truth labels for
        training.
        The following keys are expected:
          {'illuminant': the color of illuminant in RGB (an unit vector). This
            is the reciprocal of RGB gains.}
      mode: Indicates training, eval or inference mode: learn.ModeKeys.TRAIN,
        learn.ModeKeys.EVAL, learn.ModeKeys.INFER
      params: a dict with keys:
        'first_bin': (float) location of the edge of the first histogram bin.
        'bin_size': (float) size of each histogram bin.
        'nbins': (int) number of histogram bins.
        'extended_vector_length': The number of features in the extended vector.
        'ellipse_w_mat', 'elllipse_b_vec': parameters describing the ellipse of
          valid white points. See 'ellipse.py' for more details.
        'variance': a small positive value to add to the covariance matrix to
          make it definite positive.

    Returns:
      A model_fn_lib.ModelFnOps object that can be called by the estimator.
    """

    tf.compat.v1.logging.info('Run model_fn, mode=%s', mode)
    tf.compat.v1.logging.info('Params=%s', params)
    tf.compat.v1.logging.info('HParams=[%s]', hparams)

    # Expect input features is a dictionary
    assert isinstance(features, dict)
    rgb = features['rgb']
    extended_feature = features['extended_feature']
    weight = features['weight']

    n = params['nbins']
    extended_vector_length = params['extended_vector_length']
    filters_extended_latent = tf.compat.v1.get_variable(
        'filters_extended_latent',
        shape=(extended_vector_length, n * n, NUMBER_OF_CHANNELS_FILTERS),
        initializer=tf.zeros_initializer())
    filters_base_latent = tf.compat.v1.get_variable(
        'filters_base_latent',
        shape=(n * n, NUMBER_OF_CHANNELS_FILTERS),
        initializer=tf.zeros_initializer())
    bias_extended_latent = tf.compat.v1.get_variable(
        'bias_extended_latent',
        shape=(extended_vector_length, n * n),
        initializer=tf.zeros_initializer())

    # Initialize the bias to a scaled+shifted 2D Hann function in [-1, 1],
    # so that white point estimates are initially located at the center of the
    # histogram.
    hann1 = np.sin(np.pi * np.float32(np.arange(0, n)) / n)**2
    bias_base_init = 2 * hann1[np.newaxis, :] * hann1[:, np.newaxis] - 1.
    precond_bias = tf.cast(
        fft.compute_preconditioner_vec(n, hparams['mult_bias_tv'],
                                       hparams['mult_bias_l2']), tf.float32)
    bias_base_latent_init = (fft.fft2_to_vec(
        ops.r2c_fft2(
            tf.convert_to_tensor(bias_base_init)[tf.newaxis, :, :,
                                                 tf.newaxis]))[0] /
                             precond_bias)[:, 0]
    bias_base_latent = tf.compat.v1.get_variable(
        'bias_base_latent', initializer=bias_base_latent_init)

    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.math.is_nan(filters_extended_latent), False),
        tf.debugging.assert_equal(tf.math.is_nan(filters_base_latent), False),
        tf.debugging.assert_equal(tf.math.is_nan(bias_extended_latent), False),
        tf.debugging.assert_equal(tf.math.is_nan(bias_base_latent), False)
    ]):
      (filters_extended_fft, filters_base_fft, bias_extended_fft, bias_base_fft,
       bias_extended, bias_base, precond_filters,
       precond_bias) = latent_to_model(filters_extended_latent,
                                       filters_base_latent,
                                       bias_extended_latent, bias_base_latent,
                                       hparams, params)
      with tf.name_scope('preconditioner'):
        tf.summary.histogram('precond_filters', precond_filters)
        tf.summary.histogram('precond_bias', precond_bias)

      mu, sigma, heatmap = evaluate_model(rgb, extended_feature,
                                          filters_extended_fft,
                                          filters_base_fft, bias_extended,
                                          bias_base, params)

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = dict()
        predictions['uv'] = mu
        if 'name' in features:
          predictions['name'] = features['name']
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      # Computes losses
      assert isinstance(labels, dict)
      true_uv = ops.rgb_to_uv(labels['illuminant'])

      global_step = tf.compat.v1.train.get_global_step()

      # The ground-truth white points should lie within the ellipse of valid
      # white points, otherwise something has gone wrong or the ellipse needs
      # to be re-fit.
      true_ellipse_distance = ellipse.distance(
          true_uv, np.reshape(params['ellipse_w_mat'], (2, 2)),
          np.asarray(params['ellipse_b_vec']))
      with tf.control_dependencies(
          [tf.debugging.assert_less_equal(true_ellipse_distance, 1. + 1e-5)]):
        (weighted_loss_data, data_losses) = losses.compute_data_loss(
            heatmap,
            mu,
            sigma,
            true_uv,
            weight=weight,
            step_size=params['bin_size'],
            offset=params['first_bin'],
            n=n,
            rgb=rgb)

      smooth_vars = [
          filters_base_latent, filters_extended_latent, bias_base_latent,
          bias_extended_latent
      ]
      numer = tf.reduce_sum([tf.reduce_sum(v**2) for v in smooth_vars])
      denom = tf.reduce_sum(
          [tf.math.reduce_prod(tf.shape(v)) for v in smooth_vars])
      loss_smooth = numer / tf.cast(denom, tf.float32)

      loss = weighted_loss_data + hparams['mult_smooth'] * loss_smooth

      if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'rms_anisotropic_reproduction_error':
                tf.compat.v1.metrics.root_mean_squared_error(
                    predictions=data_losses['anisotropic_reproduction_error'],
                    labels=tf.zeros(
                        tf.shape(
                            data_losses['anisotropic_reproduction_error'])),
                    weights=weight),
            'mean_anisotropic_reproduction_error':
                tf.compat.v1.metrics.mean(
                    values=data_losses['anisotropic_reproduction_error'],
                    weights=weight),
            'reproduction_error':
                tf.compat.v1.metrics.mean(
                    values=data_losses['reproduction_error'], weights=weight),
            'mean_angular_error':
                tf.compat.v1.metrics.mean(
                    values=data_losses['angular_error'], weights=weight),
            'gaussian_nll':
                tf.compat.v1.metrics.mean(
                    values=data_losses['gaussian_nll'], weights=weight)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

      # Smoothly decay from 1 to 0.
      cosine_decay = tf.compat.v1.train.cosine_decay(1., global_step,
                                           hparams['total_training_iterations'])
      learning_rate = cosine_decay * hparams['learning_rate']

      assert mode == tf.estimator.ModeKeys.TRAIN
      train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(
        loss=loss, global_step=global_step)

      # Setup summaries
      with tf.name_scope('summaries'):
        # Save the total loss and anisotropic reproduction loss
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('reproduction_error',
                          tf.reduce_mean(data_losses['reproduction_error']))
        tf.summary.scalar('angular_error',
                          tf.reduce_mean(data_losses['angular_error']))
        tf.summary.scalar('loss_smooth', loss_smooth)
        tf.summary.image('filters_extended',
                         _visualize_fft(filters_extended_fft, shift=True))
        tf.summary.image('filters_base',
                         _visualize_fft(filters_base_fft, shift=True))
        tf.summary.image('bias_extended',
                         _visualize_fft(bias_extended_fft, shift=False))
        tf.summary.image('bias_base',
                         _visualize_fft(bias_base_fft, shift=False))
        tf.summary.histogram('filters_extended_latent', filters_extended_latent)
        tf.summary.histogram('filters_base_latent', filters_base_latent)
        tf.summary.histogram('bias_extended_latent', bias_extended_latent)
        tf.summary.histogram('bias_base_latent', bias_base_latent)
        tf.summary.histogram(
            'filters_extended_fft',
            tf.stack(
                [tf.math.real(filters_extended_fft),
                 tf.math.imag(filters_extended_fft)]))
        tf.summary.histogram(
            'filters_base_fft',
            tf.stack(
                [tf.math.real(filters_base_latent),
                 tf.math.imag(filters_base_latent)]))
        tf.summary.histogram(
            'bias_extended_fft',
            tf.stack(
                [tf.math.real(bias_extended_latent),
                 tf.math.imag(bias_extended_latent)]))
        tf.summary.histogram(
            'bias_base_fft',
            tf.stack([tf.math.real(bias_base_fft),
                      tf.math.imag(bias_base_fft)]))

      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  return _model_fn
