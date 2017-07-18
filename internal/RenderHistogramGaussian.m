% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function P_vis = RenderHistogramGaussian(mu, Sigma, mu_true, target_size, ...
  normalize, params, u_shift, v_shift)
% Renders a Gaussian in histogram space over a reference image showing relative
% colors. Also takes in an optional ground-truth white point to render,
% and an optional reference size to scale the visualization to.

% Compute the mahalanobis distance from the predicted white point to
% to every point in the chroma histogram.
[us, vs] = ndgrid(EnumerateBins(params));

if ~isempty(target_size)
  us = imresize(us, [target_size, target_size], 'bilinear');
  vs = imresize(vs, [target_size, target_size], 'bilinear');
end

axis_mask = (us == min(abs(us(:)))) | (vs == min(abs(vs(:))));

if exist('u_shift', 'var')
  us = us + u_shift;
end
if exist('v_shift', 'var')
  vs = vs + v_shift;
end

X = [us(:), vs(:)]';
Xc = bsxfun(@minus, X, mu);
iSigma = inv(Sigma);
Xc_iSigma = iSigma * Xc;
mahal_dist = reshape(sum(Xc .* Xc_iSigma,1), size(us));

% Threshold the mahalanobis distance at 3 to make a binary mask, which is
% dilated to produce an ellipse with a dot in the center.
mask = mahal_dist <= 3;
prediction = (conv2(double(mask), ones(3,3), 'same') > 0) & ~mask;
prediction = prediction | ...
  (conv2(double(mahal_dist == min(mahal_dist(:))), ...
  [0, 1, 0; 1, 1, 1; 0, 1, 0], 'same') > 0);

% Optionally create a mask with the ground-truth white point rendered as a dot.
if ~isempty(mu_true)
  D = (us - mu_true(1)).^2 + (vs - mu_true(2)).^2;
  truth = D == min(D(:));
  truth = (conv2(double(truth), [0, 1, 0; 1, 1, 1; 0, 1, 0], 'same') > 0);
  truth = (conv2(double(truth), [0, 1, 0; 1, 1, 1; 0, 1, 0], 'same') > 0);
else
  truth = zeros(size(us));
end

% Render a RGB image, which is white-balanced to the mean, to show the relative
% color of each histogram bin in the chroma histogram.
if normalize
  us = us - mean(us(:));
  vs = vs - mean(vs(:));
end
rgb_background = cat(3, exp(-us), ones(size(us)), exp(-vs));
rgb_background = ...
  bsxfun(@rdivide, rgb_background, max(rgb_background, [], 3));

if ~normalize
  rgb_background(~repmat(axis_mask, [1,1,3])) = ...
    rgb_background(~repmat(axis_mask, [1,1,3])) * 0.5;
end

rgb_prediction = cat(3, ones(size(us)), (1-prediction), (1-prediction));
rgb_truth = cat(3, (1-truth), (1-truth), ones(size(us)));

% Alpha matte the prediction and ground-truth over the RGB background image.
[rgb_out, alpha_out] = AlphaMatte(rgb_prediction, prediction, ...
  rgb_background, ones(size(rgb_background,1), size(rgb_background,2)));
rgb_out = AlphaMatte(rgb_truth, truth, rgb_out, alpha_out);

P_vis = uint8(round(255*rgb_out));
