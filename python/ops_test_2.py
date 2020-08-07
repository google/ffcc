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

import numpy as np
from ops import local_absolute_deviation, data_preprocess
import tensorflow.compat.v1 as tf
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import colors

SCALE_EDGES = 3

img = scipy.io.loadmat('test_data/image.mat')['img']
img = np.expand_dims(img, 0)
## testing edge image
ref_edge = scipy.io.loadmat('test_data/matlab_edge.mat')['edge']
computed_edge = local_absolute_deviation(tf.convert_to_tensor(img))
computed_edge = tf.squeeze(computed_edge).numpy()
_, ax = plt.subplots(1, 3)
ax[0].set_title('Input image')
ax[0].imshow(np.squeeze(img))
ax[1].set_title('Matlab edge')
ax[1].imshow(ref_edge * SCALE_EDGES)
ax[2].set_title('Py edge')
ax[2].imshow(computed_edge * SCALE_EDGES)
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.show()

# Check working with batches.
computed_edges = local_absolute_deviation(
    tf.concat((tf.convert_to_tensor(img), tf.convert_to_tensor(img)), axis=0))
computed_edge_1 = tf.squeeze(computed_edges[0, :, :, :]).numpy()
computed_edge_2 = tf.squeeze(computed_edges[1, :, :, :]).numpy()
_, ax = plt.subplots(1, 4)
ax[0].set_title('Input image')
ax[0].imshow(np.squeeze(img))
ax[1].set_title('Matlab edge')
ax[1].imshow(ref_edge * SCALE_EDGES)
ax[2].set_title('Py edge (1)')
ax[2].imshow(computed_edge_1 * SCALE_EDGES)
ax[3].set_title('Py edge (2)')
ax[3].imshow(computed_edge_2 * SCALE_EDGES)
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.show()

## histogram computation
ref_edge_N = scipy.io.loadmat('test_data/matlab_edge_histogram.mat')['edge_N']
ref_img_N = scipy.io.loadmat('test_data/matlab_img_histogram.mat')['img_N']
params = {"first_bin": 0.0, "bin_size": 1. / 32, "nbins": 64}
N, _ = data_preprocess(tf.convert_to_tensor(img),
                       tf.convert_to_tensor(np.ones((img.shape[0], 1))), params)
N = tf.squeeze(N, axis=0).numpy()

fig, axs = plt.subplots(2, 2)
fig.suptitle('Histograms')

images = []

axs[0, 0].set_title('Matlab image hist')
images.append(axs[0, 0].imshow(ref_img_N))
axs[0, 1].set_title('Matlab edge hist')
images.append(axs[0, 1].imshow(ref_edge_N * SCALE_EDGES))
axs[1, 0].set_title('Py image hist')
images.append(axs[1, 0].imshow(N[:, :, 0]))
axs[1, 1].set_title('Py edge hist')
images.append(axs[1, 1].imshow(N[:, :, 1] * SCALE_EDGES))

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


for im in images:
    im.callbacksSM.connect('changed', update)

plt.show()
plt.xticks([]), plt.yticks([])
plt.axis('off')

# Check working with batches
N, _ = data_preprocess(tf.concat((tf.convert_to_tensor(img),
                                  tf.convert_to_tensor(img)), axis=0),
                       tf.convert_to_tensor(np.ones((2 * img.shape[0], 1))), params)
N_1 = tf.squeeze(N[0, :, :, 0]).numpy()
N_2 = tf.squeeze(N[1, :, :, 0]).numpy()
fig, ax = plt.subplots(1, 4)
ax[0].set_title('Input image')
ax[0].imshow(np.squeeze(img))
ax[1].set_title('Matlab image hist')
ax[1].imshow(ref_img_N)
ax[2].set_title('Py hist (1)')
ax[2].imshow(N_1)
ax[3].set_title('Py hist (2)')
ax[3].imshow(N_2)
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.show()
