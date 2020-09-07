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
"""Losses for FFCC."""
import math
import sys
from ffcc import ops
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def _check_shape(rgb1, rgb2):
  rgb1.shape.assert_has_rank(2)
  rgb2.shape.assert_has_rank(2)
  rgb1.shape.assert_is_compatible_with([None, 3])
  rgb2.shape.assert_is_compatible_with([None, 3])


def safe_acosd(x):
  """Returns arccos(x) in degrees, with a "safe" and approximate gradient.

  When |x| = 1, the derivative of arccos() is infinite, and this can cause
  catastrophic problems during optimization. So though this function returns an
  accurate measure of arccos(x), the derivative it returns is that of
  arccos(0.9999999x). This should not effect the performance of FFCC learning.

  Args:
    x: an input tensor of any shape.

  Returns:
    Returns acos for each element in the unit of degrees.
  """
  # Check that x is roughly within [-1, 1].
  with tf.control_dependencies([
      tf.debugging.assert_greater_equal(x, tf.cast(-1.0 - 1e-6, dtype=x.dtype)),
      tf.debugging.assert_less_equal(x, tf.cast(1.0 + 1e-6, dtype=x.dtype))
  ]):
    x = tf.clip_by_value(x, -1.0, 1.0)
    angle_shrink = tf.acos(0.9999999 * x)
    angle_true = tf.acos(x)
    # Use angle_true for the forward pass, but use angle_shrink for backprop.
    angle = angle_shrink + tf.stop_gradient(angle_true - angle_shrink)
    return angle * (180.0 / math.pi)


def angular_error(pred_illum_rgb, true_illum_rgb):
  """Measures the angular errors of predicted illuminants and ground truth.

  Args:
    pred_illum_rgb: predicted RGB illuminants in the shape of [batch_size, 3].
    true_illum_rgb: true RGB illuminants in the shape of [batch_size, 3].

  Returns:
    Angular errors (degree) in the shape of [batch_size].
  """
  _check_shape(pred_illum_rgb, true_illum_rgb)

  pred_magnitude_sq = tf.reduce_sum(tf.square(pred_illum_rgb), axis=1)
  true_magnitude_sq = tf.reduce_sum(tf.square(true_illum_rgb), axis=1)

  with tf.control_dependencies([
      tf.debugging.assert_greater(pred_magnitude_sq,
                        tf.cast(1e-8, dtype=pred_magnitude_sq.dtype)),
      tf.debugging.assert_greater(true_magnitude_sq,
                        tf.cast(1e-8, dtype=true_magnitude_sq.dtype))
  ]):
    numer = tf.reduce_sum(tf.multiply(pred_illum_rgb, true_illum_rgb), axis=1)
    denom_sq = tf.multiply(pred_magnitude_sq, true_magnitude_sq)
    epsilon = sys.float_info.epsilon
    ratio = (numer + epsilon) * tf.compat.v1.rsqrt(denom_sq + epsilon * epsilon)
    return safe_acosd(ratio)


def reproduction_error(pred_illum_rgb, true_illum_rgb):
  """Measures the reproduction errors of predicted illuminants and ground truth.

  An implementation of "Reproduction Angular Error", as described in
  "Reproduction Angular Error: An Improved Performance Metric for Illuminant
  Estimation", Graham Finlayson and Roshanak Zakizadeh, BMVC 2014.
  http://www.bmva.org/bmvc/2014/papers/paper047/

  Args:
    pred_illum_rgb: predicted RGB illuminants in the shape of [batch_size, 3].
    true_illum_rgb: true RGB illuminants in the shape of [batch_size, 3].

  Returns:
    Reproduction angular errors (degree) in the shape of [batch_size].
  """
  _check_shape(pred_illum_rgb, true_illum_rgb)

  epsilon = sys.float_info.epsilon
  ratio = true_illum_rgb / (pred_illum_rgb + epsilon)
  numer = tf.reduce_sum(ratio, axis=1)
  denom_sq = 3 * tf.reduce_sum(ratio**2, axis=1)
  angle_prod = (numer + epsilon) * tf.compat.v1.rsqrt(
    denom_sq + epsilon * epsilon)
  return safe_acosd(angle_prod)


def anisotropic_reproduction_error(pred_illum_rgb, true_illum_rgb,
                                   true_scene_rgb):
  """Measures anisotropic reproduction error wrt the average scene color.

  The output error is invariant to the absolute scale of all inputs.

  Args:
    pred_illum_rgb: predicted RGB illuminants in the shape of [batch_size, 3].
    true_illum_rgb: true RGB illuminants in the shape of [batch_size, 3].
    true_scene_rgb: averaged scene RGB of the true white-balanced image in the
      shape of [batch_size, 3].

  Returns:
    Anisotropic reproduction angular errors (degree) in the shape of
    [batch_size].
  """
  _check_shape(pred_illum_rgb, true_illum_rgb)
  _check_shape(pred_illum_rgb, true_scene_rgb)

  epsilon = sys.float_info.epsilon
  ratio = true_illum_rgb / (pred_illum_rgb + epsilon)
  numer = tf.reduce_sum(true_scene_rgb**2 * ratio, axis=1)
  denom_sq = tf.reduce_sum(
      true_scene_rgb**2, axis=1) * tf.reduce_sum(
          true_scene_rgb**2 * ratio**2, axis=1)
  angle_prod = (numer + epsilon) * tf.compat.v1.rsqrt(
    denom_sq + epsilon * epsilon)
  return safe_acosd(angle_prod)


def anisotropic_reproduction_loss(pred_illum_uv, true_illum_uv, rgb):
  """Computes anisotropic reproduction loss.

  Args:
    pred_illum_uv: float, predicted uv in log-UV space, in the shape of
      [batch_size, 2].
    true_illum_uv: the true white points in log-UV space, in the shape of
      [batch_size, 2].
    rgb: float, the input RGB image that the log-UV chroma histograms are
      constructed from, [batch_size, height, width, channels].

  Returns:
    float, weighted repdocution errors in the shape of [batch_size].
  """
  pred_illum_uv.shape.assert_is_compatible_with([None, 2])
  true_illum_uv.shape.assert_is_compatible_with([None, 2])
  rgb.shape.assert_is_compatible_with([None, None, None, 3])

  pred_illum_rgb = ops.uv_to_rgb(pred_illum_uv)
  true_illum_rgb = ops.uv_to_rgb(true_illum_uv)
  true_scene_rgb = tf.reduce_mean(ops.apply_wb(rgb, true_illum_uv), axis=[1, 2])
  return anisotropic_reproduction_error(pred_illum_rgb, true_illum_rgb,
                                        true_scene_rgb)


def gaussian_negative_log_likelihood(pred_illum_uv, pred_illum_uv_sigma,
                                     true_illum_uv):
  """Computes the negative log-likelihood of a multivariate gaussian.

  This implements the loss function described in the FFCC paper, Eq. (18).

  Args:
    pred_illum_uv: float, predicted uv in log-UV space, in the shape of
      [batch_size, 2].
    pred_illum_uv_sigma: float, the predicted covariance matrix for
      `pred_illum_uv`, with a shape of [batch_size, 2, 2].
    true_illum_uv: the true white points in log-UV space, in the shape of
      [batch_size, 2].

  Returns:
    The negative log-likelihood of the multivariate Gaussian distribution
    defined by pred_illum_uv (mu) and pred_illum_uv_sigma (sigma), evaluated at
    true_illum_uv.
  """
  det = tf.linalg.det(pred_illum_uv_sigma)
  with tf.control_dependencies(
      [tf.debugging.assert_greater(det, tf.cast(0.0, dtype=det.dtype))]):
    pred_pdf = tfp.distributions.MultivariateNormalFullCovariance(
        loc=pred_illum_uv, covariance_matrix=pred_illum_uv_sigma)
  return -pred_pdf.log_prob(true_illum_uv)


def compute_data_loss(pred_heatmap, pred_illum_uv, pred_illum_uv_sigma,
                      true_illum_uv, weight, step_size, offset, n, rgb):
  """Computes the data term of the loss function.

  The data loss is the (squared) anisotropic reproduction error.

  This function also returns the unweighted, un-squared, sub-loss, which
  is used for summaries and for reporting training / eval errors.

  Args:
    pred_heatmap: float, the network predictions in the shape of [batch_size, n,
      n].
    pred_illum_uv: float, the predicted white point in log-UV space in the shape
      of [batch_size, 2].
    pred_illum_uv_sigma: float, the predicted covariance matrix for
      `pred_illum_uv`, with a shape of [batch_size, 2, 2].
    true_illum_uv: float, the true white point in log-UV space in the shape of
      [batch_size, 2].
    weight: float, the weight for the loss in the shape of [batch_size]
    step_size: float, the pitch of each step, scalar.
    offset: float, the value of the first index, scalar.
    n: float, the number of bins, scalar.
    rgb: float, the original input RGB thumbnails in the shape of [batch_size,
      height, width, 3].

  Returns:
    A tuple of the form:
    weighted_loss_data, a float containing the total weighted loss
    losses, a dict with keys:
      'anisotropic_reproduction_error': vector of floats containing the
        anisotropic reproduction error for each datapoint in the batch.
      'angular_error': vector of floats containing the angular error for each
        datapoint in the batch.
  """
  pred_heatmap.shape.assert_is_compatible_with([None, n, n])
  pred_illum_uv.shape.assert_is_compatible_with([None, 2])
  pred_illum_uv_sigma.shape.assert_is_compatible_with([None, 2, 2])
  true_illum_uv.shape.assert_is_compatible_with([None, 2])
  if not np.isscalar(step_size):
    raise ValueError('`step_size` must be a scalar, but is of type {}'.format(
        type(step_size)))
  if not np.isscalar(offset):
    raise ValueError('`step_size` must be a scalar, but is of type {}'.format(
        type(offset)))
  if not np.isscalar(n):
    raise ValueError('`n` must be a scalar, but is of type {}'.format(type(n)))
  rgb.shape.assert_is_compatible_with([None, None, None, 3])

  losses = {
      'anisotropic_reproduction_error':
          anisotropic_reproduction_loss(pred_illum_uv, true_illum_uv, rgb),
      'reproduction_error':
          reproduction_error(
              ops.uv_to_rgb(pred_illum_uv), ops.uv_to_rgb(true_illum_uv)),
      'angular_error':
          angular_error(
              ops.uv_to_rgb(pred_illum_uv), ops.uv_to_rgb(true_illum_uv)),
      'gaussian_nll':
          gaussian_negative_log_likelihood(pred_illum_uv, pred_illum_uv_sigma,
                                           true_illum_uv)
  }

  # We minimize the (weighted) NLL loss for training.
  weighted_loss_data = tf.reduce_mean(weight * losses['gaussian_nll'])

  with tf.name_scope('data'):
    # Render out the PMF (red), the predicted UV coordinate from fitting a
    # Von Mises to that PMF (green), and the true UV coordinate.
    # UV coordinates are rendered by splatting them to a histogram.
    tf.summary.histogram('pred_heatmap', pred_heatmap)

    pred_pmf = ops.softmax2(pred_heatmap)

    def _normalize(x):
      ep = sys.float_info.epsilon
      max_val = tf.reduce_max(x, [1, 2], keepdims=True) + ep
      return x / max_val

    vis_pred_cross = _normalize(pred_pmf)
    vis_pred_aniso = _normalize(
        ops.uv_to_pmf(pred_illum_uv, step_size, offset, n))
    vis_true = _normalize(ops.uv_to_pmf(true_illum_uv, step_size, offset, n))
    vis = tf.cast(
        tf.round(255 *
                 tf.stack([vis_pred_cross, vis_pred_aniso, vis_true], axis=-1)),
        tf.uint8)
    tf.summary.image('pmf_argmax_gt', vis)

    def _make_montage(v, k=4):
      """Attempt to make a k*k montage of `v`, if the batch size allows it."""
      count = tf.minimum(
          k,
          tf.cast(
              tf.floor(tf.sqrt(tf.cast(tf.shape(v)[0], tf.float32))), tf.int32))
      montage = v[:(count**2), :, :, :]
      montage = tf.reshape(
          tf.transpose(
              tf.reshape(montage, [count, count, v.shape[1], v.shape[2], 3]),
              [0, 2, 1, 3, 4]), (count * v.shape[1], count * v.shape[2], 3))
      return montage

    montage = _make_montage(vis)
    tf.summary.image('pmf_argmax_gt_montage', montage[tf.newaxis])

    uv_range = np.arange(n) * step_size + offset
    vv, uu = np.meshgrid(uv_range, uv_range)
    uv = np.stack([uu, vv], axis=-1)
    pred_pdf = tf.transpose(
        tfp.distributions.MultivariateNormalFullCovariance(
            loc=pred_illum_uv,
            covariance_matrix=pred_illum_uv_sigma).prob(uv[:, :,
                                                           tf.newaxis, :]),
        [2, 0, 1])
    vis_pred_pdf = _normalize(pred_pdf)
    vis_pdf = tf.cast(
        tf.round(255 *
                 tf.stack([vis_pred_cross, vis_pred_pdf, vis_true], axis=-1)),
        tf.uint8)
    tf.summary.image('pmf_pdf_gt', vis_pdf)

    montage_pdf = _make_montage(vis_pdf)
    tf.summary.image('pmf_pdf_gt_montage', montage_pdf[tf.newaxis])

  return (weighted_loss_data, losses)
