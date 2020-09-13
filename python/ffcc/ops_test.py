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
"""Tests for ops.py."""

import random

from ffcc import ops
import numpy as np
from scipy import signal
import tensorflow as tf
import tensorflow_probability as tfp



class OpsTest(tf.test.TestCase):

  def _eval(self, tensor, feed_dict=None):
    with tf.compat.v1.Session() as sess:
      return sess.run(tensor, feed_dict=feed_dict)

  def setUp(self):
    super(OpsTest, self).setUp()
    np.random.seed(0)

  def testFft2RoundTrip(self):
    # I == ifft2(fft2(I))
    batch_size = 2
    t = np.random.uniform(size=(batch_size, 3, 7, 1))
    t_reconstructed = np.asarray(
        self._eval(
            ops.c2r_ifft2(ops.r2c_fft2(tf.constant(t, dtype=tf.float32)))))
    np.testing.assert_allclose(t, t_reconstructed, atol=1e-06)

  def testEvalFeaturesWithDeltaFunctions(self):
    """Tests EvalFeatures with delta functions.

    Given a constant 2-channel matrix of 0.5, with a delta function (as identity
    matrix). The returned result should a single-channel all-one image,
    e.g.: 2*f = ifft(fft(2*f))
    """

    batch_size = 2
    height = 3
    width = 3
    channels = 2
    features = np.ones((batch_size, height, width, channels)) * 0.5
    kernel = np.zeros((batch_size, height, width, channels))
    kernel[:, 1, 1, 0] = 1.
    kernel[:, 1, 1, 1] = 1.
    kernel_fft = ops.r2c_fft2(tf.constant(kernel, tf.float32))

    bias = np.random.uniform(size=(batch_size, height, width))
    h = np.asarray(
        self._eval(
            ops.eval_features(
                tf.constant(features, dtype=tf.float32), kernel_fft, bias)))
    np.testing.assert_allclose(h, np.sum(features, axis=-1) + bias)

  def testEvalFeaturesWithCircularConv2(self):
    """Tests EvalFeatures with 2D circular convolution.

    Given a random single channel matrix, X, and a random single channel filter,
    F: ifft2(fft2(X) .* fft2(F)) == conv2d(X, F, boundary='wrap')
    """

    batch_size = 2
    height = 3
    width = 3
    features = np.random.uniform(size=(batch_size, height, width, 1))
    kernel = np.random.uniform(size=(batch_size, height, width, 1))
    kernel_fft = ops.r2c_fft2(tf.constant(kernel, tf.float32))
    zero_bias = np.zeros((batch_size, height, width))
    h = np.asarray(
        self._eval(
            ops.eval_features(
                tf.constant(features, dtype=tf.float32), kernel_fft,
                zero_bias)))

    for i in range(batch_size):
      h_ref = signal.convolve2d(
          np.squeeze(features[i, :, :, :]),
          np.squeeze(kernel[i, :, :, :]),
          mode='full',
          boundary='wrap')
      np.testing.assert_allclose(
          h[i, :, :], h_ref[:height, :width], rtol=1e-05, atol=1e-07)

  def testSoftmax2(self):
    """Check the simple 2D case where are dimensions are softmax'ed."""
    batch_size = 2
    dimension = 3
    scale = random.random()
    h = np.exp(
        np.random.uniform(size=(batch_size, dimension, dimension)) * scale)

    p = self._eval(ops.softmax2(tf.constant(h, dtype=tf.float32)))

    # p should sum to 1.
    np.testing.assert_allclose(np.sum(p, axis=(1, 2)), 1, rtol=1e-5)

    # softmax2 should be shift-invariant.
    for _ in range(20):
      h2 = h + random.random() * 100
      p2 = self._eval(ops.softmax2(tf.constant(h2, dtype=tf.float32)))
      np.testing.assert_allclose(p2, p, rtol=1e-5)

  def testBivariateVonMises(self):
    """Tests Bivariate von Mises.

    Compute the histogram with givan mean and covariance matrix and see if the
    returned mean is within the tolerated difference.
    """
    n_batch = 10
    n_bins = 64
    n_random = 8

    # Generate the indices for a grid.
    vv, uu = np.meshgrid(range(n_bins), range(n_bins))
    uv = np.stack([uu, vv], axis=-1)

    mus_true = []
    sigmas_true = []
    pmfs = []
    for _ in range(n_batch):
      # Generate random data near the center of the histogram, and get its
      # mean, covariance, and PMF.
      x = np.random.normal(
          loc=n_bins / 2., scale=n_bins / 16., size=(2, n_random))
      mu_true = np.mean(x, axis=1)
      sigma_true = np.cov(x)
      pmf_true = tfp.distributions.MultivariateNormalFullCovariance(
          loc=mu_true, covariance_matrix=sigma_true).prob(uv)
      pmf_true /= tf.reduce_sum(pmf_true)

      # Shift the mean and the PMF by some integer value.
      int_shift = np.int32(np.random.uniform(low=-n_bins, high=n_bins, size=2))
      mu_true = np.mod(mu_true + int_shift, n_bins)
      pmf_true = tf.roll(tf.roll(pmf_true, int_shift[0], 0), int_shift[1], 1)

      mus_true.append(mu_true)
      sigmas_true.append(sigma_true)
      pmfs.append(pmf_true)
    mus_true = tf.stack(mus_true, axis=0)
    sigmas_true = tf.stack(sigmas_true, axis=0)
    pmfs = tf.stack(pmfs, axis=0)

    # Fit a mean and covariance to each PMF.
    mus, sigmas = ops.bivariate_von_mises(tf.cast(pmfs, tf.float32))

    self.assertAllClose(mus, mus_true, atol=1e-5, rtol=1e-5)
    self.assertAllClose(sigmas, sigmas_true, atol=1e-3, rtol=1e-3)

  def testIdxToUv(self):
    """Tests ops.idx_to_uv."""
    batch_size = 3
    step_size = 1. / 32.
    mu_indices = np.random.randint(low=0, high=5, size=(batch_size, 2))
    sigma_indices = np.random.randint(low=0, high=5, size=(batch_size, 2, 2))
    offset = np.random.uniform(size=(2))

    mu_ops, sigma_ops = ops.idx_to_uv(
        tf.constant(mu_indices, dtype=tf.int32),
        tf.constant(sigma_indices, dtype=tf.int32), step_size,
        tf.constant(offset, dtype=tf.float32))

    np.testing.assert_allclose(
        np.asarray(self._eval(mu_ops)), mu_indices * step_size + offset)
    np.testing.assert_allclose(
        np.asarray(self._eval(sigma_ops)), sigma_indices * (step_size**2))

  def testSplatNonUniform(self):
    """Tests ops.splat_non_uniform."""
    xs = np.asarray([0.5, 1, 1.5, 2, 2.5, 4, 4.5, 8, 8.5])
    bins = np.asarray([1, 2, 4, 8])

    f = ops.splat_non_uniform(xs, bins)

    # Checks expected result
    # pyformat: disable
    np.testing.assert_allclose(
        f,
        np.asarray([[1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0.5, 0.5, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0.75, 0.25, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0.875, 0.125],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1]]))
    # pyformat: enable

    # Random xs that are within the range.
    batch_size = 5
    xs = np.random.uniform(low=bins[0], high=bins[-1], size=(batch_size))
    f = ops.splat_non_uniform(xs, bins)
    np.testing.assert_allclose(xs, np.sum(f * bins[np.newaxis, :], axis=1))

  def testUvToPmf(self):
    offset = 0.5
    step_size = 1
    n = 3

    # pyformat: disable
    uvs = [[0.5, 0.5],
           [1.0, 0.5],
           [0.5, 1.0],
           [1.0, 1.0],
           [2.5, 2.5],
           [1.5, 1.5],
           [1.5, 2.0],
           [2.0, 1.5],
           [2.0, 2.0],
           [8.0, 1.5],
           [-8.0, 1.5],
           [1.5, 8.0],
           [1.5, -8.0]]

    expected_pmfs = [[[1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]],
                     [[0.5, 0.0, 0.0],
                      [0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0]],
                     [[0.5, 0.5, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]],
                     [[0.25, 0.25, 0.0],
                      [0.25, 0.25, 0.0],
                      [0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.5, 0.0]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.25, 0.25],
                      [0.0, 0.25, 0.25]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]],
                     [[0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]]]
    # pyformat: enable

    actual_pmfs = self._eval(
        ops.uv_to_pmf(tf.constant(uvs), step_size, offset, n))
    # Checks if the PMFs match the expectation.
    np.testing.assert_equal(expected_pmfs, actual_pmfs)

  def testRgbToUvAndBack(self):
    batch_size = 100
    rgb = np.random.uniform(size=(batch_size, 3))

    # normalized RGB values to unit vectors
    rgb /= np.linalg.norm(rgb, axis=1, keepdims=True)
    np.testing.assert_almost_equal(
        rgb, self._eval(ops.uv_to_rgb(ops.rgb_to_uv(tf.constant(rgb)))))

  def testApplyWb(self):
    batch_size = 100
    width = 64
    height = 48
    channels = 3
    rgbs = np.random.uniform(size=(batch_size, height, width, channels))
    wb_gains = np.random.uniform(low=0.1, high=1.0, size=(batch_size, channels))
    wb_gains /= np.expand_dims(wb_gains[:, 1], axis=1)

    expected_wb_rgbs = rgbs * wb_gains[:, np.newaxis, np.newaxis, :]
    np.testing.assert_almost_equal(
        expected_wb_rgbs,
        self._eval(
            ops.apply_wb(
                tf.constant(rgbs), ops.rgb_to_uv(tf.constant(1.0 / wb_gains)))))


if __name__ == '__main__':
  tf.test.main()
