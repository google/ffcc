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

function [X, im_channels] = FeaturizeImage(im, mask, params)
% Given an input RGB image and a binary mask indicating which pixels are to
% be trusted in that image, produces 2D histogram of the log-chroma of the
% image.

if isempty(mask)
  mask = true(size(im,1), size(im,2));
end

im_channels = ChannelizeImage(im, mask);

if isa(im, 'float')
  assert(all(im(:) <= 1));
  assert(all(im(:) >= 0));
end

X = {};
for i_channel = 1:length(im_channels)

  im_channel = im_channels{i_channel};

  log_im_channel = {};
  for c = 1:size(im_channel, 3)
    log_im_channel{c} = log(double(im_channel(:,:,c)));
  end
  u = log_im_channel{2} - log_im_channel{1};
  v = log_im_channel{2} - log_im_channel{3};

  % Masked pixels or those with invalid log-chromas (nan or inf) are
  % ignored.
  valid = ~isinf(u) & ~isinf(v) & ~isnan(u) & ~isnan(v) & mask;

  % Pixels whose intensities are less than a (scaled) minimum_intensity are
  % ignored. This enables repeatable behavior for different input types,
  % otherwise we see behavior where the input type affects output features
  % strongly just by how intensity values get quantized to 0.
  if isa(im, 'float')
    min_val = params.HISTOGRAM.MINIMUM_INTENSITY;
  else
    min_val = intmax(class(im)) * params.HISTOGRAM.MINIMUM_INTENSITY;
  end
  valid = valid & all(im_channel >= min_val, 3);

  Xc = Psplat2(u(valid), v(valid), ones(nnz(valid),1), ...
    params.HISTOGRAM.STARTING_UV, params.HISTOGRAM.BIN_SIZE, ...
    params.HISTOGRAM.NUM_BINS);

  Xc = Xc / max(eps, sum(Xc(:)));

  X{end+1} = Xc;
end

X = cat(3, X{:});

end
