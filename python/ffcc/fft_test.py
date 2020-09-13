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
"""Tests for fft.py."""
from ffcc import fft
import numpy as np
from scipy import signal
import tensorflow as tf


class FftTest(tf.test.TestCase):

  def setUp(self):
    super(FftTest, self).setUp()
    np.random.seed(0)

  def testExceptionOnOddSizeFft(self):
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        fft.fft2_to_vec(tf.constant(np.fft.fft2(np.random.randn(1, 3, 3, 1))))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        fft.vec_to_fft2(tf.constant(np.random.randn(1, 3 * 3, 1)))

  def testExceptionOnNonSquaredFft(self):
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        fft.fft2_to_vec(tf.constant(np.fft.fft2(np.random.randn(1, 3, 2, 1))))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        fft.vec_to_fft2(tf.constant(np.random.randn(1, 3 * 2, 1)))

  def testForwardAndInverse(self):
    batch_size = 100
    channels = 10

    for n in [2, 4, 6, 8]:
      m = np.fft.fft2(np.random.randn(batch_size, n, n, channels), axes=[1, 2])
      self.assertAllClose(fft.vec_to_fft2(fft.fft2_to_vec(tf.constant(m))), m)

  def testMagnitudePreservation(self):
    n = 64
    batch_size = 100
    channels = 10

    m = np.fft.fft2(np.random.randn(batch_size, n, n, channels), axes=[1, 2])
    m_squared_norm = np.sum(np.abs(m)**2, axis=(1, 2))

    self.assertAllClose(
        tf.reduce_sum(fft.fft2_to_vec(tf.constant(m))**2, axis=1),
        m_squared_norm)

  def testComputeRegularizerFft(self):
    n = 64
    num_channels = 10
    x = np.random.randn(n, n, num_channels)

    # Compute the regularizer.
    weight_tv = np.random.uniform(size=num_channels)
    weight_l2 = np.random.uniform(size=num_channels)
    reg = (fft.compute_regularizer_fft(n, weight_tv, weight_l2))

    # Compute the loss according to the regularizer.
    x_fft = np.fft.fft2(x, axes=[0, 1])
    loss = np.sum(reg * (np.real(x_fft)**2 + np.imag(x_fft)**2)) / (n**2)

    # Compute the loss on the pixels directly.
    loss_true = 0.
    for channel in range(num_channels):
      xc = x[:, :, channel]
      grad_x = signal.convolve2d(xc, [[-1, 1]], mode='same', boundary='wrap')
      grad_y = signal.convolve2d(xc, [[-1], [1]], mode='same', boundary='wrap')
      loss_true += (
          weight_l2[channel] * np.sum(xc**2) + weight_tv[channel] *
          (np.sum(grad_x**2) + np.sum(grad_y**2)))

    self.assertAllClose(loss, loss_true)

  def testPreconditionerMatchesRegularizer(self):
    # Note: This test depends on the correctness of compute_regularizer_fft().
    n = 8
    num_channels = 10
    x = np.random.randn(n, n, num_channels)
    x_fft = np.fft.fft2(x, axes=[0, 1])

    # Compute the regularizer and preconditioner.
    weight_tv = np.random.uniform(size=num_channels)
    weight_l2 = np.random.uniform(size=num_channels)

    # Compute the loss according to the regularizer.
    reg = fft.compute_regularizer_fft(n, weight_tv, weight_l2)
    loss_true = np.sum(reg * (np.real(x_fft)**2 + np.imag(x_fft)**2)) / (n**2)

    # Compute the loss according to the preconditioner.
    precond = fft.compute_preconditioner_vec(n, weight_tv, weight_l2)
    x_vec = fft.fft2_to_vec(tf.constant(x_fft)[tf.newaxis])[0, :, :]
    x_vec_precond = x_vec / precond
    loss = tf.reduce_sum(x_vec_precond**2) / (n**2)

    self.assertAllClose(loss, loss_true)


if __name__ == '__main__':
  tf.test.main()
