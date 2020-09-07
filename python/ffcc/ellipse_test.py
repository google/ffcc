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
"""Tests for ellipse.py."""

from ffcc import ellipse
import numpy as np
from scipy.linalg import expm
import tensorflow as tf


class EllipseTest(tf.test.TestCase):

  def _get_random_ellipse(self, dim):
    x = np.random.uniform(size=(dim, dim))
    w_mat = expm((np.transpose(x) + x) / 2.0)
    b_vec = np.random.uniform(size=(dim))
    return w_mat, b_vec

  def _eval(self, tensor, feed_dict=None):
    with tf.Session() as sess:
      return sess.run(tensor, feed_dict=feed_dict)

  def setUp(self):
    super(EllipseTest, self).setUp()
    np.random.seed(0)

  def testStandardToGeneralSpecialCases(self):
    """Sanity-check a few easy to verify cases of standard_to_general()."""
    a_mat = np.float32(np.array([[1, 0], [0, 1]]))
    c_vec = np.float32(np.array([0, 0]))
    w_mat, b_vec = ellipse.standard_to_general(a_mat, c_vec)
    self.assertAllClose(w_mat, a_mat)
    self.assertAllClose(b_vec, c_vec)

    a_mat = np.float32(np.array([[1, 0], [0, 1]]))
    c_vec = np.float32(np.random.normal(size=2))
    w_mat, b_vec = ellipse.standard_to_general(a_mat, c_vec)
    self.assertAllClose(w_mat, a_mat)
    self.assertAllClose(b_vec, -c_vec)

    a_mat = np.float32(np.array([[2., 0.], [0., 0.5]]))
    c_vec = np.float32(np.array([1., 1.]))
    w_mat, b_vec = ellipse.standard_to_general(a_mat, c_vec)
    self.assertAllClose(w_mat, np.array([[np.sqrt(2), 0], [0, np.sqrt(0.5)]]))
    self.assertAllClose(b_vec, np.array([-np.sqrt(2), -np.sqrt(0.5)]))

  def testGeneraltoStandardRoundTrip(self):
    for _ in range(10):
      w_mat, b_vec = self._get_random_ellipse(2)
      a_mat, c_vec = ellipse.general_to_standard(w_mat, b_vec)
      w_mat_recon, b_vec_recon = ellipse.standard_to_general(a_mat, c_vec)
      self.assertAllClose(w_mat, w_mat_recon)
      self.assertAllClose(b_vec, b_vec_recon)

  def testStandardToGeneralDistancesMatch(self):
    """Check distance() against the standard parametrization's distance."""
    for _ in range(10):
      num_dims = 2
      w_mat, b_vec = self._get_random_ellipse(num_dims)
      a_mat, c_vec = ellipse.general_to_standard(w_mat, b_vec)
      data = np.random.normal(size=(100, num_dims))
      dist_general = ellipse.distance(data, w_mat, b_vec)
      dist_standard = tf.reduce_sum(
          (data - c_vec) * tf.linalg.matmul((data - c_vec), a_mat), axis=-1)
      self.assertAllClose(dist_general, dist_standard)

  def testProject(self):
    for _ in range(10):
      dim = np.random.randint(low=1, high=10)
      w_mat, b_vec = self._get_random_ellipse(dim)

      batch_size = 100
      x = np.random.normal(size=(batch_size, dim))
      proj = self._eval(ellipse.project(x, w_mat, b_vec))
      x_distance = self._eval(ellipse.distance(x, w_mat, b_vec))
      proj_distance = self._eval(ellipse.distance(proj, w_mat, b_vec))
      self.assertTrue(np.all(np.less_equal(proj_distance, 1.0 + 1e-8)))

      # Check points with distance < 1 have not changed.
      mask = x_distance < 1
      self.assertTrue(np.all(np.equal(x[mask, :], proj[mask, :])))

      # Check points with distance >= 1 have a distance of 1 after projection.
      np.testing.assert_allclose(proj_distance[x_distance >= 1], 1.)

      # Check that the projected points are scaled versions of the input points.
      center = -np.matmul(np.linalg.inv(w_mat), b_vec)
      delta_ratio = (x - center) / (proj - center)
      avg_delta_ratio = np.tile(
          np.mean(delta_ratio, axis=-1, keepdims=True),
          (1, delta_ratio.shape[1]))
      np.testing.assert_allclose(delta_ratio, avg_delta_ratio)


if __name__ == '__main__':
  tf.test.main()
