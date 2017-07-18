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

function VisASCII(im, equalize_histogram)
% A helper function for visualizing images as ASCII text. For each 2D image in
% the n-channel input image, renders out an ASCII image from a small library of
% "grayscale" characters. This can be helpful when debugging over SSH.
% If equalize_histogram == true then the image will be histogram-equalized
% before being rendered.

if nargin < 2
  equalize_histogram = false;
end

% These characters correspond to: ' ░▒▓█'
mapping = [' ', char(9617), char(9618), char(9619), char(9608)]

if equalize_histogram
  im = double(im);
  for c = 1:size(im,3)
    [~, ~, u] = unique(reshape(im(:, :, c), [], 1));
    im(:,:,c) = reshape(u, size(im(:,:,c)));
  end
end

% Normalize and rescale the image.
im = im - min(im(:));
im = im ./ (eps + max(im(:)));
im = uint8(1 + min(length(mapping)-1, floor(im * length(mapping))));

% Render out each channel of the image.
for c = 1:size(im,3)
  for i = 1:size(im,1)
    fprintf('%s\n', mapping(im(i,:,c)));
  end
  fprintf('\n');
end
