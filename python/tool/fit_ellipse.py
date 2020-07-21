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
r"""Fit black body ellipse from the training data.

The visualization will be saved under /tmp/ellipse_fit.png
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from . import ellipse
from . import input as ffcc_input
from . import ops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
tf.app.flags.DEFINE_string('data_dir', '', 'Input dir for the training data.')

FLAGS = tf.app.flags.FLAGS


def export_csv(filename, burst_id, uv, d):
  """Export CSV file.

  Args:
    filename: the filename of the CSV file.
    burst_id: the list of burst id associate with the data.
    uv: the list of log-UV coordinates of the data.
    d: the list of ellipse distance of the data.
  """
  with open(filename, 'w') as csvfile:
    filewriter = csv.writer(
        csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['burst_id', 'log(g/r)', 'log(g/b)', 'd'])
    for i in range(len(burst_id)):
      filewriter.writerow([burst_id[i], uv[i, 0], uv[i, 1], d[i]])


def fit_ellipse(data, num_iters=2500):
  """Fits an ellipse the `data` matrix."""
  # Compute the convex hull of the UV coordinates.
  hull_idx = scipy.spatial.ConvexHull(data).vertices
  hull = data[hull_idx, :]

  # Fit a covariance matrix to the convex hull points. This covariance matrix
  # will determine the orientation of the axes of the ellipse, and the relative
  # scaling of the major and minor axes.
  sigma = np.float32(np.cov(np.transpose(hull)))

  # Find the center of the ellipse, by projecting the convex hull points into
  # the eigenvectors of the covariance matrix and finding the centroid of the
  # minimum and maximum projected points.
  _, eig_vecs = np.linalg.eig(sigma)
  proj = np.matmul(hull, eig_vecs)
  max_val = np.amax(proj, axis=0)
  min_val = np.amin(proj, axis=0)
  c_vec = np.matmul(eig_vecs, (max_val + min_val) / 2.)

  # Scale the quadratic that defines the ellipse (the inverse of the convariance
  # matrix) around the center point to exactly contain the points within the
  # convex hull.
  hull_centered = hull - c_vec
  a_mat = np.linalg.inv(sigma)
  a_mat /= np.max(
      np.sum(hull_centered * (np.matmul(hull_centered, a_mat)), axis=1))

  # Here we refine all parameters of the ellipse shape to make it as tight as
  # possible, by optimizing over the matrix log to maximize its determinant
  # while also fitting all points within it (approximated using quadratic hinge
  # loss on all points outside of the ellipse).
  # This is almost certainly suboptimal, but it's hard to derive a tight
  # analytical solution to this problem.

  # To ensure that a_mat is PSD, we reparametrize it as "half" of its matrix
  # log, so a_mat = expm(half_log_a_mat + half_log_a_mat.T)
  log_a_mat = tf.cast(tf.linalg.logm(tf.cast(a_mat, tf.complex64)), tf.float32)
  triu_log_a_mat = tf.linalg.band_part(log_a_mat, 0, -1)
  half_log_a_mat = tf.linalg.set_diag(triu_log_a_mat,
                                      tf.linalg.diag_part(triu_log_a_mat) * 0.5)
  half_log_a_mat = tf.Variable(half_log_a_mat, dtype=tf.float32)
  c_vec = tf.Variable(c_vec, dtype=tf.float32)
  optimizer = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
  variables = [half_log_a_mat, c_vec]
  ii = 0
  while True:
    with tf.GradientTape() as tape:
      a_mat = tf.linalg.expm(half_log_a_mat + tf.transpose(half_log_a_mat))
      w_mat, b_vec = ellipse.standard_to_general(a_mat, c_vec)
      dist = ellipse.distance(hull, w_mat, b_vec)
      loss_data = tf.reduce_sum(tf.maximum(0., dist - 1.))
      loss_tight = -tf.math.log(tf.linalg.det(a_mat))
      loss = loss_data + loss_tight
    grads = tape.gradient(loss, variables)
    if ii % 100 == 0:
      tf.logging.info('%05d: %0.8f = %e + %0.8f', ii, loss, loss_data,
                      loss_tight)
    # Terminate when the number of iterations is over the threshold and the data
    # term of the loss is zero, at which point optimization has hopefully
    # converged and all points are guaranteed to be inside the ellipse.
    if loss_data <= 0. and ii > num_iters:
      tf.logging.info('%05d: %0.8f = %e + %0.8f, terminating (%d > %d)', ii,
                      loss, loss_data, loss_tight, ii, num_iters)
      break
    optimizer.apply_gradients(zip(grads, variables))
    ii += 1
  a_mat = a_mat.numpy()
  c_vec = c_vec.numpy()

  # Scale the ellipse's A matrix to be as tight as possible.
  hull_centered = hull - c_vec
  a_mat /= np.max(
      np.sum(hull_centered * (np.matmul(hull_centered, a_mat)), axis=1))

  # The determinant of the ellipse's A matrix is proportional to the tightness
  # of the fit of the ellipse. We print out the inverse determinant to check if
  # fit is reasonable.
  goodness = np.linalg.det(a_mat)
  tf.logging.info('fit = %f', goodness)

  # We check that all ground-truth points lie within the ellipse.
  data_centered = data - c_vec
  np.testing.assert_almost_equal(
      np.amax(np.sum(data_centered * np.matmul(data_centered, a_mat), axis=1)),
      1.,
      decimal=5)

  return a_mat, c_vec


def main(_):
  if not tf.gfile.IsDirectory(FLAGS.data_dir):
    tf.logging.error('Invalid input directory: %s', FLAGS.data_dir)
    return

  tf.logging.info('Loading dataset')
  (train_input_fn, _, _, _, _) = (
      ffcc_input.input_builder_stratified(
          FLAGS.data_dir, batch_size=1, num_epochs=1, bucket_size=1))

  # Extract the ground-truth UV coordinates of the illuminants.
  dataset = train_input_fn()
  items = []
  for (feature, label) in dataset:
    items.append({
        'name': feature['name'],
        'uv': ops.rgb_to_uv(label['illuminant']),
    })

  ids = [np.asscalar(item['name'].numpy()) for item in items]
  uvs = np.float32(np.concatenate([item['uv'].numpy() for item in items]))

  a_mat, c_vec = fit_ellipse(uvs)

  # Change the ellipse's parametrization.
  w_mat, b_vec = ellipse.standard_to_general(a_mat, c_vec)

  # Sanity check that all datapoints lie within the ellipse.
  d = ellipse.distance(uvs, w_mat, b_vec)
  np.testing.assert_array_less(d, 1. + 1e-5)

  # Sanity check that projection is a no-op.
  uvs_projected = ellipse.project(uvs, w_mat, b_vec)
  np.testing.assert_almost_equal(uvs_projected, uvs, decimal=5)

  # Generate the ellipse distance to the CSV file.
  export_csv('/tmp/ellipse_distance.csv', ids, np.asarray(uvs), np.asarray(d))

  # Draw ellipse.
  # Compute "tilt" of ellipse using first eigenvector.
  eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(a_mat))
  x, y = eig_vecs[:, 0]
  theta = np.degrees(np.arctan2(y, x))

  # Eigenvalues give length of ellipse along each eigenvector.
  w, h = 2 * np.sqrt(eig_vals)
  fig, ax = plt.subplots(1)
  ax.set_xlabel('log(g/r)')
  ax.set_ylabel('log(g/b)')
  ax.scatter(x=uvs[:, 0], y=uvs[:, 1], marker='+', color='r')
  patch = patches.Ellipse(c_vec, w, h, theta, color='g')
  patch.set_clip_box(ax.bbox)
  patch.set_alpha(0.2)
  ax.add_artist(patch)
  tf.logging.info('saving figures ')
  fig.savefig('/tmp/ellipse_fit.png')

  tf.logging.info('ellipse alpha = %s', 1e8)
  tf.logging.info('w_mat = %s', w_mat.numpy().reshape([-1]))
  tf.logging.info('b_vec = %s', b_vec.numpy())


if __name__ == '__main__':
  tf.app.run(main)
