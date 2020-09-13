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
"""Tests for losses.py."""
import copy
import math
import sys

from ffcc import losses
from ffcc import ops
import numpy as np
import tensorflow as tf


def _normalized(x):
  return x / np.linalg.norm(x, axis=1, keepdims=True)


def _weighted_normalized(x, weight):
  return _normalized(x * weight)


def _generate_products(num_samples):
  # Generate `num_samples` values in [-1, 1], concentrated near the edges.
  x = np.linspace(-1, 1, num_samples, np.float32)
  x = np.sign(x) * (1. - np.square(1. - np.abs(x)))
  return x


def _random_covariance_matrix(batch_size):
  """Generate a batch of random covariance matrices.

  Args:
    batch_size: Number of elements in the first dimension returned.

  Returns:
    A tensor with dimensions [batch_size, 2, 2].
  """
  # Make a random covariance matrix by taking the outer product of 10 random
  # matrices.
  x = np.random.normal(size=(batch_size, 10, 2))
  sigma = np.matmul(x.transpose(0, 2, 1), x)
  return sigma


class SafeArcCosDTest(tf.test.TestCase):
  """Tests losses.safe_acosd()."""

  def testFiniteAngleAndGradient(self):
    """Test that the angle and the gradient are finite for all x in [-1, 1]."""
    x = _generate_products(100000)
    with self.session() as sess:
      # Query the angle and the derivative at all points.
      x_ph = tf.compat.v1.placeholder(x.dtype, len(x))
      angle_ph = losses.safe_acosd(x_ph)
      angle, (d_angle) = sess.run(
          (angle_ph, tf.gradients(tf.reduce_sum(angle_ph), x_ph)),
          feed_dict={x_ph: x})
      # Verify that all angles and derivatives are finite.
      for v in [angle, d_angle]:
        self.assertTrue(np.all(np.isfinite(v)))

  def testAccurateAngle(self):
    """Test that the angle is very accurate."""
    x = _generate_products(10000)
    # Query the angle at all points.
    angle = losses.safe_acosd(x)
    angle_true = np.arccos(x) * 180. / math.pi
    self.assertAllClose(angle, angle_true)

  def testAccurateDerivative(self):
    """Test that the derivative is mostly accurate."""
    # Ignore the extreme values of x.
    x = _generate_products(10000)[100:-100]
    with self.session() as sess:
      # Query the derivative at all points.
      x_ph = tf.compat.v1.placeholder(x.dtype, len(x))
      d_angle = sess.run(
          tf.gradients(tf.reduce_sum(losses.safe_acosd(x_ph)), x_ph),
          feed_dict={x_ph: x})[0]
    d_angle_true = -180. / (math.pi * np.sqrt(1. - np.square(x)))
    # Test with a loose relative tolerance, as magnitudes will be large if |x|
    # is large.
    self.assertAllClose(d_angle, d_angle_true, atol=1e-6, rtol=1e-3)


class AngularErrorTest(tf.test.TestCase):
  """Tests losses.angular_error."""

  def testOrthogonalVectors(self):
    """Tests two orthogonal vectors as input."""
    pred_illum_rgb = tf.constant(
        np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
    true_illum_rgb = tf.constant(
        np.asarray([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]))
    self.assertAllClose(
        losses.angular_error(pred_illum_rgb, true_illum_rgb),
        np.repeat(90,
                  pred_illum_rgb.get_shape().as_list()[0]))

  def testAntiParallelVectors(self):
    """Tests two antiparallel vectors as input."""
    batch_size = 100
    pred_illum_rgb = tf.constant(np.random.randn(batch_size, 3))
    true_illum_rgb = -pred_illum_rgb
    self.assertAllClose(
        losses.angular_error(pred_illum_rgb, true_illum_rgb),
        np.repeat(180, batch_size))

  def testAgainstIdenticalVectors(self):
    """Tests two identical vectors as input."""
    batch_size = 100
    rgb = tf.constant(np.random.randn(batch_size, 3))
    self.assertAllClose(
        losses.angular_error(rgb, rgb), np.repeat(0., batch_size), atol=1e-5)

    # Should be scale invariant
    self.assertAllClose(
        losses.angular_error(rgb, rgb * 10.),
        np.repeat(0., batch_size),
        atol=1e-5)

  def testIllegalZeroMagnitudeVectors(self):
    """Tests one of the inputs is a zero vector."""
    batch_size = 100
    rgb = tf.constant(np.random.randn(batch_size, 3))
    zeros = tf.constant(np.zeros(shape=(batch_size, 3)))

    with self.session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        losses.angular_error(rgb, zeros).eval()

    with self.session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        losses.angular_error(zeros, rgb).eval()

    with self.session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        losses.angular_error(zeros, zeros).eval()

  def testAgainstRefImpl(self):
    """Tests against reference implementation."""
    batch_size = 100
    pred_illum_rgb = np.random.randn(batch_size, 3)
    true_illum_rgb = np.random.randn(batch_size, 3)

    expected_angle = (180. / math.pi) * np.arccos(
        np.sum(
            _normalized(pred_illum_rgb) * _normalized(true_illum_rgb), axis=1))

    self.assertAllClose(
        losses.angular_error(
            tf.constant(pred_illum_rgb), tf.constant(true_illum_rgb)),
        expected_angle)


class ReproductionErrorTest(tf.test.TestCase):
  """Tests losses.reproduction_error."""

  def testAgainstRefImpl(self):
    """Tests against reference implementation."""
    batch_size = 100
    pred_illum_rgb = np.random.rand(batch_size, 3)
    true_illum_rgb = np.random.rand(batch_size, 3)

    expected_angle = (180. / math.pi) * np.arccos(
        np.sum(
            _normalized(true_illum_rgb / pred_illum_rgb) *
            _normalized(np.ones(shape=(1, 3))),
            axis=1))

    self.assertAllClose(
        losses.reproduction_error(
            tf.constant(pred_illum_rgb), tf.constant(true_illum_rgb)),
        expected_angle)

  def testWhiteIlluminant(self):
    """Tests against white (neutral gray) illuminants."""
    # The reproduction angular error with respect to white (a vector of ones)
    # should be equal to the angular error between 1/the estimated illuminant
    # and a vector of ones.
    batch_size = 100
    pred_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_illum_rgb = tf.constant(np.ones(shape=(batch_size, 3)))

    self.assertAllClose(
        losses.reproduction_error(pred_illum_rgb, true_illum_rgb),
        losses.angular_error(tf.compat.v1.reciprocal(pred_illum_rgb), true_illum_rgb))

  def testTintInvariant(self):
    """Tests tint invariant property."""
    batch_size = 100
    pred_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_illum_rgb = tf.constant(np.ones(shape=(batch_size, 3)))
    tint = tf.constant(np.random.rand(1, 3))
    self.assertAllClose(
        losses.reproduction_error(pred_illum_rgb, true_illum_rgb),
        losses.reproduction_error(
            tf.multiply(tint, pred_illum_rgb),
            tf.multiply(tint, true_illum_rgb)))


class AnisotropicReproductionErrorTest(tf.test.TestCase):
  """Tests losses.anisotropic_reproduction_error."""

  def testConstantWeight(self):
    """Tests constant weights."""
    batch_size = 100
    pred_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_scene_rgb = tf.constant(
        np.exp(np.random.rand(batch_size, 1)) * np.ones(shape=(batch_size, 3)))

    self.assertAllClose(
        losses.anisotropic_reproduction_error(pred_illum_rgb, true_illum_rgb,
                                              true_scene_rgb),
        losses.reproduction_error(pred_illum_rgb, true_illum_rgb))

  def testAgainstRefImpl(self):
    """Tests against reference implementation."""
    batch_size = 100
    pred_illum_rgb = np.random.rand(batch_size, 3)
    true_illum_rgb = np.random.rand(batch_size, 3)
    true_scene_rgb = np.random.rand(batch_size, 3)

    expected_angle = (180. / math.pi) * np.arccos(
        np.clip(
            np.sum(
                _weighted_normalized(true_illum_rgb / pred_illum_rgb,
                                     true_scene_rgb) *
                _weighted_normalized(np.ones(shape=(1, 3)), true_scene_rgb),
                axis=1), -1, 1))
    self.assertAllClose(
        losses.anisotropic_reproduction_error(
            tf.constant(pred_illum_rgb), tf.constant(true_illum_rgb),
            tf.constant(true_scene_rgb)),
        expected_angle,
        atol=1e-5)

  def testAgainstIdenticalIluminations(self):
    """Tests the case where predicted ilumminants is as same as ground truth."""
    batch_size = 100
    pred_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_scene_rgb = tf.constant(np.random.rand(batch_size, 3))

    # The errors should drop to zeros.
    self.assertAllClose(
        losses.anisotropic_reproduction_error(pred_illum_rgb, pred_illum_rgb,
                                              true_scene_rgb),
        np.zeros(batch_size),
        atol=1e-5)

  def testNonNegativeScaleInvariant(self):
    """Tests invariant property wrt (non-negative) scale of true_scene_rgb."""
    batch_size = 100
    pred_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_scene_rgb = tf.constant(np.random.rand(batch_size, 3))
    scale = tf.constant(np.exp(np.random.rand(batch_size, 1)))

    self.assertAllClose(
        losses.anisotropic_reproduction_error(pred_illum_rgb, true_illum_rgb,
                                              true_scene_rgb),
        losses.anisotropic_reproduction_error(
            pred_illum_rgb, true_illum_rgb, tf.multiply(scale, true_scene_rgb)))

    self.assertAllClose(
        losses.anisotropic_reproduction_error(pred_illum_rgb, true_illum_rgb,
                                              true_scene_rgb),
        losses.anisotropic_reproduction_error(
            pred_illum_rgb, tf.multiply(scale, true_illum_rgb), true_scene_rgb))

    self.assertAllClose(
        losses.anisotropic_reproduction_error(pred_illum_rgb, true_illum_rgb,
                                              true_scene_rgb),
        losses.anisotropic_reproduction_error(
            tf.multiply(scale, pred_illum_rgb), true_illum_rgb, true_scene_rgb))

  def testZeroedWeightsSingleChannel(self):
    """Tests one of the true_scene_rgb is zero."""
    batch_size = 100
    pred_illum_rgb = np.random.rand(batch_size, 3)
    true_illum_rgb = np.random.rand(batch_size, 3)
    true_scene_rgb = np.exp(np.random.rand(batch_size, 3))

    for c in range(3):
      zeroed_true_scene_rgb = copy.deepcopy(true_scene_rgb)
      zeroed_true_scene_rgb[:, c] = 0

      true_illum_rgb_adjusted = copy.deepcopy(true_illum_rgb)
      true_illum_rgb_adjusted[:, c] = true_illum_rgb_adjusted[:, c] + 0.1

      # The error should be invariant on the channel with zero weight.
      self.assertAllClose(
          losses.anisotropic_reproduction_error(
              tf.constant(pred_illum_rgb), tf.constant(true_illum_rgb),
              tf.constant(zeroed_true_scene_rgb)),
          losses.anisotropic_reproduction_error(
              tf.constant(pred_illum_rgb), tf.constant(true_illum_rgb_adjusted),
              tf.constant(zeroed_true_scene_rgb)))

  def testZeroedWeightsDualChannels(self):
    """Tests two of the true_scene_rgb are zero."""
    batch_size = 100
    pred_illum_rgb = np.random.rand(batch_size, 3)
    true_illum_rgb = np.random.rand(batch_size, 3)
    true_scene_rgb = np.exp(np.random.rand(batch_size, 3))

    for c in range(3):
      zeroed_true_scene_rgb = np.zeros(shape=(batch_size, 3))
      zeroed_true_scene_rgb[:, c] = true_scene_rgb[:, c]

      # The errors should drop to zeros.
      self.assertAllClose(
          losses.anisotropic_reproduction_error(
              tf.constant(pred_illum_rgb), tf.constant(true_illum_rgb),
              tf.constant(zeroed_true_scene_rgb)),
          np.zeros(batch_size),
          atol=1e-5)

  def testTintInvariant(self):
    """Tests tint invariant property."""
    batch_size = 100
    pred_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_illum_rgb = tf.constant(np.random.rand(batch_size, 3))
    true_scene_rgb = tf.constant(np.exp(np.random.rand(batch_size, 3)))
    tint = tf.constant(np.random.rand(1, 3))
    self.assertAllClose(
        losses.anisotropic_reproduction_error(pred_illum_rgb, true_illum_rgb,
                                              true_scene_rgb),
        losses.anisotropic_reproduction_error(
            tf.multiply(tint, pred_illum_rgb),
            tf.multiply(tint, true_illum_rgb), true_scene_rgb))


class AnisotropicReproductionLossTest(tf.test.TestCase):
  """Tests losses.anisotropic_reproduction_loss."""

  def testAgainstRefImpl(self):
    batch_size = 100
    pred_illum_rgb = np.random.uniform(
        low=sys.float_info.epsilon, size=(batch_size, 3))
    pred_illum_rgb /= np.sum(pred_illum_rgb, axis=1, keepdims=True)

    true_illum_rgb = np.random.uniform(
        low=sys.float_info.epsilon, size=(batch_size, 3))
    true_illum_rgb /= np.sum(true_illum_rgb, axis=1, keepdims=True)
    true_wb_rgb = 1.0 / true_illum_rgb
    true_wb_rgb /= np.expand_dims(true_wb_rgb[:, 1], axis=1)

    true_scene_rgb = np.random.uniform(
        low=sys.float_info.epsilon, size=(batch_size, 3))
    input_scene_rgb = true_scene_rgb / true_wb_rgb

    height = 64
    width = 48
    rgb_stats = np.tile(
        input_scene_rgb[:, np.newaxis, np.newaxis, :],
        reps=[1, height, width, 1])

    with self.test_session() as sess:
      actual_result = sess.run(
          losses.anisotropic_reproduction_loss(
              ops.rgb_to_uv(tf.constant(pred_illum_rgb)),
              ops.rgb_to_uv(tf.constant(true_illum_rgb)),
              tf.constant(rgb_stats)))
      expected_result = sess.run(
          losses.anisotropic_reproduction_error(
              tf.constant(pred_illum_rgb), tf.constant(true_illum_rgb),
              tf.constant(true_scene_rgb)))
      np.testing.assert_almost_equal(actual_result, expected_result)


class GaussianNegativeLogLikelihoodTest(tf.test.TestCase):
  """Tests losses.gaussian_negative_log_likelihood."""

  def testBasicCorrectness(self):
    """Test with ground truth values equal to predictions."""
    batch_size = 100
    # Generate random covariance matrices. Add a small epsilon on the diagonal
    # to make them definite positive.
    sigma = tf.eye(2, batch_shape=(batch_size,))
    rgb = np.random.uniform(
        low=sys.float_info.epsilon, size=(batch_size, 3)).astype('float32')
    uv = ops.rgb_to_uv(rgb)
    # If ground truth values are equal to predictions, and the matrix 'sigma'
    # is constant (over the batch size), the result will be a constant as well.
    actual_result = losses.gaussian_negative_log_likelihood(uv, sigma, uv)
    expected_result = tf.repeat(actual_result[0], batch_size)
    self.assertAllClose(expected_result, actual_result)


class ComputeDataLossTest(tf.test.TestCase):
  """Tests losses.compute_data_loss."""

  def setUp(self):
    super(ComputeDataLossTest, self).setUp()
    batch_size = 100
    n = 64
    channels = 4
    width = 64
    height = 48

    pred_pmf = np.random.uniform(size=(batch_size, n, n))
    pred_pmf = pred_pmf / np.sum(pred_pmf, axis=(1, 2), keepdims=True)

    # Generate a random covariance matrix by taking the outer product of 10
    # random matrices.
    sigma = _random_covariance_matrix(batch_size)

    self._pred_pmf = pred_pmf
    self._pred_uv = np.random.uniform(size=(batch_size, 2))
    self._pred_uv_sigma = sigma
    self._true_illum_uv = np.random.uniform(size=(batch_size, 2))
    self._ones = np.ones(batch_size)
    self._weights = np.random.uniform(size=(batch_size))
    self._step_size = 0.3125
    self._offset = -1.0
    self._n = n
    self._channels = channels
    self._rgb_stats = np.random.uniform(size=(batch_size, height, width, 3))

  def _run(self):
    return losses.compute_data_loss(
        tf.constant(self._pred_pmf), tf.constant(self._pred_uv),
        tf.constant(self._pred_uv_sigma), tf.constant(self._true_illum_uv),
        tf.constant(self._weights), self._step_size, self._offset, self._n,
        tf.constant(self._rgb_stats))[0]

  def testVonMisesAnisotropicReproductionAlone(self):
    with self.session() as sess:
      actual_result = self._run()

      expected_result = tf.reduce_mean(
          losses.gaussian_negative_log_likelihood(
              tf.constant(self._pred_uv), tf.constant(self._pred_uv_sigma),
              tf.constant(self._true_illum_uv)))

      np.testing.assert_almost_equal(
          sess.run(actual_result), sess.run(expected_result))


if __name__ == '__main__':
  tf.test.main()
