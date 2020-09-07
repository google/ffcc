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
"""FFT utility functions."""

import math
import numpy as np
import tensorflow as tf


def fftshift(f, axis):
  """Shift zero-frequency component to center of spectrum.

  Args:
    f: float or imaginary tensor.
    axis: tuple, the axes to be shifted.

  Returns:
    A center-shifted tensor in the same shape of f.
  """
  shift = [f.shape.as_list()[ax] // 2 for ax in axis]
  return tf.roll(f, shift=shift, axis=axis)


def fft2_to_vec(f):
  """Converts a 2D n-by-n Fourier coefficients to a 1D real vector.

  This function implements Real Bijective FFT for 2D squared FFT where the size
  has to be even number. More detail in:
    Fast Fourier Color Constancy, Barron and Tsai, CVPR 2017:
    https://arxiv.org/abs/1611.07596

  Limitation: n has to be even size.

  Args:
    f: Stacked of multi-channel 2D n-by-n Fourier coefficients in the shape of
      [batch_size, n, n, channels], where each 2D coefficients are serialized
      into 1D vector independently.

  Returns:
    1D real vector in the shape of [batch_size, n*n, channels].
  """
  f_real = tf.math.real(f)
  f_imag = tf.math.imag(f)

  _, height, width, channels = f.get_shape().as_list()
  deps = [tf.debugging.assert_equal(width, height),
          tf.debugging.assert_equal(width % 2, 0)]

  with tf.control_dependencies(deps):
    n = width
    s = n // 2
    scaling = math.sqrt(2)

    batch_size = tf.shape(f)[0]

    # The four elements that only appear once in the entire 2D FFT.
    mask = np.zeros(shape=(n, n), dtype=bool)
    mask[[0, 0, s, s], [0, s, 0, s]] = True
    f_real = tf.where(
        tf.tile(mask[tf.newaxis, :, :, tf.newaxis],
                [batch_size, 1, 1, channels]), f_real / scaling, f_real)

    v = tf.concat([
        f_real[:, 0:s+1, 0, :],
        f_real[:, 0:s+1, s, :],
        tf.reshape(f_real[:, :, 1:s, :],
                   [batch_size, n * (s - 1), channels]),
        f_imag[:, 1:s, 0, :],
        f_imag[:, 1:s, s, :],
        tf.reshape(f_imag[:, :, 1:s, :],
                   [batch_size, n * (s - 1), channels])
    ], axis=1)  # pyformat: disable

    return v * scaling


def vec_to_fft2(v):
  """Inverse of fft2_to_vec.

  This function implements Real Bijective FFT for 2D squared FFT where the size
  has to be even number. More detail in:
    Fast Fourier Color Constancy, Barron and Tsai, CVPR 2017:
    https://arxiv.org/abs/1611.07596

  Limitation: v has to be the length of n*n where n is an even number.

  Args:
    v: 1D real vector represents 2D n-by-n Fourier coefficients in the shape of
      [batch_size, n*n, channels].

  Returns:
    2D n-by-n Fourier coefficients in the shape of [batch_size, n, n, channels].
  """
  _, n_square, channels = v.get_shape().as_list()
  n = int(math.sqrt(n_square))

  dtype = tf.math.real(v).dtype
  deps = [tf.debugging.assert_equal(n * n, n_square),
          tf.debugging.assert_equal(n % 2, 0)]
  with tf.control_dependencies(deps):
    batch_size = tf.shape(v)[0]

    s = n // 2
    t = n * (s - 1) + 2 * (s + 1)
    # Constructs the slice [:, :, 0, :]
    f_real_slice0 = tf.concat(
        [v[:, 0:s + 1, :], tf.reverse(v[:, 1:s, :], axis=[1])], axis=1)

    zeros = tf.zeros((batch_size, 1, channels), dtype=dtype)
    f_imag_slice0 = tf.concat([
        zeros, v[:, t:t + s - 1, :], zeros,
        tf.reverse(-v[:, t:t + s - 1, :], axis=[1])
    ],
                              axis=1)
    f_complex_slice0 = tf.complex(f_real_slice0,
                                  f_imag_slice0)[:, :, tf.newaxis, :]

    # Constructs the slice [:, :, s, :]
    f_real_slice1 = tf.concat([
        v[:, s + 1:2 * (s + 1), :],
        tf.reverse(v[:, s + 2:2 * s + 1, :], axis=[1])
    ],
                              axis=1)
    f_imag_slice1 = tf.concat([
        zeros, v[:, t + s - 1:t + n - 2, :], zeros,
        tf.reverse(-v[:, t + s - 1:t + n - 2, :], axis=[1])
    ],
                              axis=1)
    f_complex_slice1 = tf.complex(f_real_slice1,
                                  f_imag_slice1)[:, :, tf.newaxis, :]

    # Constructs the slice [:, :, 1:s, :]
    f_real_slice2 = tf.reshape(v[:, 2 * (s + 1):2 * (s + 1) + (n * (s - 1)), :],
                               [batch_size, n, s - 1, channels])
    f_imag_slice2 = tf.reshape(v[:, t + n - 2:t + n - 2 + n * (s - 1)],
                               [batch_size, n, s - 1, channels])
    f_complex_slice2 = tf.complex(f_real_slice2, f_imag_slice2)

    # Constructs the slice [:, :, 0:s+1, :]
    f_complex_left = tf.concat(
        [f_complex_slice0, f_complex_slice2, f_complex_slice1], axis=2)

    # Constructs the slice [:, :, s+1:, :]
    f_complex_right = tf.concat([
        tf.reverse(tf.compat.v1.conj(f_complex_left[:, 0:1, 1:s, :]), axis=[2]),
        tf.reverse(tf.compat.v1.conj(f_complex_left[:, 1:, 1:s, :]), axis=[1, 2])
    ],
                                axis=1)

    f_complex_full = tf.concat([f_complex_left, f_complex_right], axis=2)

    # The four elements only appear once in the entire 2D FFT.
    scaling = math.sqrt(2)
    mask = np.zeros(shape=(n, n), dtype=bool)
    mask[[0, 0, s, s], [0, s, 0, s]] = True
    f_complex_full = tf.where(
        tf.tile(mask[tf.newaxis, :, :, tf.newaxis],
                [batch_size, 1, 1, channels]), f_complex_full * scaling,
        f_complex_full)

    return f_complex_full / scaling


def compute_regularizer_fft(n, weight_tv, weight_l2):
  """Precompute 2D filter regularizer (total variation + L2) in Fourier domain.

  This function implements w^2 in eq. 23 of the paper:
  Fast Fourier Color Constancy, Barron and Tsai, CVPR 2017
  https://arxiv.org/abs/1611.07596

  Args:
    n: specifies the square filter size in one of the dimensions.
    weight_tv: weight for the total variation term, can be a list or scalar.
    weight_l2: weight for the l2 term, can be a list of scalar.

  Returns:
    A numpy array containing an n-by-n regularizer for FFTs, with a shape of
      [n, n, channel]. The channel size is max(len(weight_tv), len(weight_l2)).
  """
  weight_tv = np.atleast_1d(weight_tv)
  weight_l2 = np.atleast_1d(weight_l2)

  magnitude = lambda x: np.real(x)**2 + np.imag(x)**2
  grad_squared_x = magnitude(np.fft.fft2([[-1, 1]], s=(n, n)))
  grad_squared_y = magnitude(np.fft.fft2([[-1], [1]], s=(n, n)))
  grad_squared = grad_squared_x + grad_squared_y
  regularizer = (grad_squared[:, :, np.newaxis] * weight_tv + weight_l2)
  return regularizer


def compute_preconditioner_vec(n, weight_tv, weight_l2):
  """Computes a Jacobi preconditioner in "vectorized" space.

  This function implements 1/w in eq. 23 of the paper:
  Fast Fourier Color Constancy, Barron and Tsai, CVPR 2017
  https://arxiv.org/abs/1611.07596

  Args:
    n: specifies the square filter size in one of the dimensions.
    weight_tv: weight for the total variation term, can be a list or scalar.
    weight_l2: weight for the l2 term, can be a list of scalar.

  Returns:
    A tensorflow array containing an Jacobi preconditioner for FFTs, with a
      shape of [n*n, max(len(weight_tv), len(weight_l2))].
  """

  regularizer = compute_regularizer_fft(n, weight_tv, weight_l2)
  scaling = np.sqrt(np.sqrt(2.))
  preconditioner = scaling * tf.compat.v1.rsqrt(
      fft2_to_vec(tf.complex(regularizer, regularizer)[tf.newaxis])[0, :, :])
  s = n // 2
  mask = np.zeros(shape=(n**2), dtype=bool)
  mask[[0, s, s + 1, n + 1]] = True
  preconditioner = tf.where(
      tf.tile(mask[:, tf.newaxis], [1, regularizer.shape[2]]),
      preconditioner / scaling, preconditioner)
  return preconditioner
