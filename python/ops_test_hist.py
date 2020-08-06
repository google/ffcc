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
from ops import masked_local_absolute_deviation, data_preprocess
import tensorflow.compat.v1 as tf
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import colors


SACLE_EDGES = 3

img = scipy.io.loadmat('test_data/image.mat')['img']
img = np.expand_dims(img, 0)
## testing edge image
ref_edge = scipy.io.loadmat('test_data/matlab_edge.mat')['edge']
computed_edge, mask_edge = masked_local_absolute_deviation(
    tf.convert_to_tensor(img))
computed_edge = tf.squeeze(computed_edge).numpy()
_, ax = plt.subplots(1, 3)
ax[0].set_title('Input image')
ax[0].imshow(np.squeeze(img))
ax[1].set_title('Matlab edge')
ax[1].imshow(ref_edge * SACLE_EDGES)
ax[2].set_title('Py edge')
ax[2].imshow(computed_edge * SACLE_EDGES)
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.show()


## histogram computation
ref_edge_N = scipy.io.loadmat('test_data/matlab_edge_histogram.mat')['edge_N'] * SACLE_EDGES
ref_img_N = scipy.io.loadmat('test_data/matlab_img_histogram.mat')['img_N']
params = {"first_bin": 0.0, "bin_size": 1./32, "nbins": 64}
N, _ = data_preprocess(tf.convert_to_tensor(img),
                    tf.convert_to_tensor(np.ones((img.shape[0], 1))), params)
N = tf.squeeze(N, axis=0).numpy()


fig, axs = plt.subplots(2, 2)
fig.suptitle('Histograms')

images = []

axs[0, 0].set_title('Matlab image hist')
images.append(axs[0, 0].imshow(ref_img_N))
axs[0, 1].set_title('Matlab edge hist')
images.append(axs[0, 1].imshow(ref_edge_N))
axs[1, 0].set_title('Py image hist')
images.append(axs[1, 0].imshow(N[:, :, 0]))
axs[1, 1].set_title('Py edge hist')
images.append(axs[1, 1].imshow(N[:, :, 1] * SACLE_EDGES))

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

