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
"""Ellipse utilities."""
import tensorflow as tf


def _check_inputs(mat, vec):
  """Checks the matrix and vector describing an ellipse (in either form)."""
  mat = tf.convert_to_tensor(mat)
  vec = tf.convert_to_tensor(vec)
  if len(mat.shape) != len(vec.shape) + 1:
    raise ValueError(
        'rank(`mat`) is not one more than rank(`vec`) ({} != {}+1)'.format(
            len(mat.shape), len(vec.shape)))
  if mat.shape.as_list()[-1] != vec.shape.as_list()[-1]:
    raise ValueError(
        '`mat`s outermost dimension ({}) does not mat `vec`s ({}))'.format(
            mat.shape.as_list()[-1],
            vec.shape.as_list()[-1]))
  if mat.shape.as_list()[-1] != mat.shape.as_list()[-2]:
    raise ValueError('`mat`s last two dimensions ({}, {}) do not match)'.format(
        mat.shape.as_list()[-1],
        mat.shape.as_list()[-2]))


def general_to_standard(w_mat, b_vec):
  """Converts a quadratic from "general form" to "standard form".

  That is, a quadratic given in the form of
    (`w_mat` * `x` + `b_vec`)^2
  is converted to a quadratic in the form of
    (`x` - `c_vec`)^T a_mat (`x` - `c_vec`)

  Args:
    w_mat: the shear of the quadratic.
    b_vec: the shift of the quadratic.

  Returns:
    A tuple containing:
      (a_mat: the "inverse covariance" of the quadratic,
       c_vec: the "center" of the quadratic)
  """
  _check_inputs(w_mat, b_vec)
  a_mat = tf.linalg.matmul(w_mat, w_mat)
  c_vec = -tf.linalg.matvec(tf.linalg.inv(w_mat), b_vec)
  return a_mat, c_vec


def standard_to_general(a_mat, c_vec):
  """Converts a quadratic from "standard form" to "general form".

  That is, a quadratic given in the form of
    (`x` - `c_vec`)^T `a_mat` (`x` - `c_vec`)
  is converted to a quadratic in the form of
    (`w_mat` * `x` + `b_vec`)^2

  Args:
    a_mat: the "inverse covariance" of the quadratic.
    c_vec: the "center" of the quadratic.

  Returns:
    A tuple containing:
      (w_mat: the shear of the quadratic,
       b_vec: the shift of the quadratic)
  """
  _check_inputs(a_mat, c_vec)
  w_mat = tf.linalg.sqrtm(a_mat)
  b_vec = -tf.linalg.matvec(w_mat, c_vec)
  return w_mat, b_vec


def distance(x, w_mat, b_vec):
  """Returns distance w.r.t an ellipse.

  The ellipse distance is defined as:
    sum((`w_mat` * `x` + `b_vec`).^2)

  Args:
    x: an input real tensor of in the shape of [batch_size, dim], where each
      x[i, :] is a vector for which distance is computed.
    w_mat: matrix representation of a conic section, in the shape of [dim, dim].
    b_vec: vector representation of a b_vec from the origin, in the shape of
      [dim].

  Returns:
    An output tensor in the shape of [batch_size] that corresponds to the
    ellipse distance of each x[i, :].
  """
  _check_inputs(w_mat, b_vec)
  v = tf.linalg.matmul(x, tf.cast(w_mat, x.dtype), transpose_b=True) + b_vec
  return tf.reduce_sum(tf.square(v), axis=-1)


def project(x, w_mat, b_vec):
  """Given an input point x, projects it onto the ellipse.

  The ellipse is defined by: sum((`w_mat` * `x` + `b_vec`).^2) <= 1

  Args:
    x: an input tensor in the shape of [batch_size, dim], where each x[i, :] is
      a vector to be projected.
    w_mat: matrix representation of a conic section, in the shape of [dim, dim].
    b_vec: vector with an offset from the origin, in the shape of [dim].

  Returns:
    An output tensor in the same shape of x that represents projected vector.
  """
  _check_inputs(w_mat, b_vec)

  d = distance(x, w_mat, b_vec)
  scale = tf.compat.v1.rsqrt(tf.maximum(d, 1))
  _, c_vec = general_to_standard(w_mat, b_vec)
  y = scale[:, tf.newaxis] * x + (1 - scale[:, tf.newaxis]) * tf.cast(
      c_vec[tf.newaxis], x.dtype)
  return y
