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

function image_down = ReadSRGBImage(im_filename, params)
% Read in an 8-bit SRGB image and preprocess it according to the stats that
% are expected by the project's params struct.

image = imread(im_filename);

% Mask out pixels if any value is above the prescribed threshold.
mask = all(image < params.HISTOGRAM.SRGB_SATURATION_VALUE, 3);
image = double(image) / 255;
image(repmat(~mask, [1,1,3])) = 0;

% If the image and the stats have inverted aspect ratios, rotate the
% image and mask.
stats_size = params.SENSOR.STATS_SIZE;
if (sign(log(size(image,1) / size(image,2))) ...
    ~= sign(log(stats_size(1) / stats_size(2))))
  image = cat(3, ...
    rot90(image(:,:,1)), ...
    rot90(image(:,:,2)), ...
    rot90(image(:,:,3)));
  mask = rot90(mask);
end

% Crop off the last row/column of the image so its size is divisible by 2,
% if necessary.
image = image(...
  1:floor(size(image,1)/2)*2, ...
  1:floor(size(image,2)/2)*2, :);

mask = mask(...
  1:floor(size(mask,1)/2)*2, ...
  1:floor(size(mask,2)/2)*2, :);

% Crop the image/mask to match the aspect ratio of the specified stats.
im_size = [size(image,1), size(image,2)];
scale = min(im_size ./ stats_size);
crop_size = floor( (stats_size * scale)/2 ) * 2;
image = image(...
  (im_size(1) - crop_size(1))/2 + [1:crop_size(1)], ...
  (im_size(2) - crop_size(2))/2 + [1:crop_size(2)], :);
mask = mask(...
  (im_size(1) - crop_size(1))/2 + [1:crop_size(1)], ...
  (im_size(2) - crop_size(2))/2 + [1:crop_size(2)]);
im_size = [size(image,1), size(image,2)];

% Check that the image and the stats have the same aspect ratio.
assert(abs(log(im_size(2) / im_size(1)) ...
  - log(crop_size(2) / crop_size(1))) < 0.01)

% Downsample the image according to the mask, and downsample the mask.
% This downsample is weighted (or equivalently, done in homogenous
% coordinates) so that masked pixels are ignored.
downsample = @(x)imresize(x, stats_size, 'bilinear');

image_numer = downsample(bsxfun(@times, image, mask));
image_denom = downsample(double(mask));
image_down = bsxfun(@rdivide, image_numer, max(eps, image_denom));

% A small denominator means that very few masked pixels are in the
% full-res image at that position, and so the low-res image should be
% masked out at that position.
DENOM_THRESHOLD = 0.01;
mask_down = image_denom >= DENOM_THRESHOLD;

% Zero out the masked values.
image_down(repmat(~mask_down, [1,1,3])) = 0;

% Convert the image to the specified bit depth, and cast accordingly.
bit_width = params.SENSOR.STATS_BIT_DEPTH;
image_down = round((2^bit_width-1) * image_down);
if bit_width <= 8
  image_down = uint8(image_down);
elseif bit_width <= 16
  image_down = uint16(image_down);
else
  assert(0);
end
