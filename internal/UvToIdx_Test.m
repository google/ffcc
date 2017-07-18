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

function UvToIdx_Test

% Test that values within the histogram range correctly map to their
% nearest bin.
for iter = 1:100

  params.HISTOGRAM.NUM_BINS = ceil(rand*100);
  params.HISTOGRAM.BIN_SIZE = 0.5 + rand;
  params.HISTOGRAM.STARTING_UV = ...
    randn - params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE/2;

  bins = EnumerateBins(params);

  Y = max(bins(1), min(bins(end), randn(2, 100) * 10));
  Y_idx = UvToIdx(Y, params);

  [~, min_idx1] = min(abs(bsxfun(@minus, Y(1,:), bins')), [], 1);
  [~, min_idx2] = min(abs(bsxfun(@minus, Y(2,:), bins')), [], 1);
  Y_idx_ = [min_idx1; min_idx2];

  assert(all(Y_idx_(:) == Y_idx(:)));

  % Check that when we map back from index to UV value, we are within a
  % half-binwidth.
  % TODO(barron): At some point we should have IdxToUv(), and that should be
  % used here.
  Y_ = (Y_idx-1) * params.HISTOGRAM.BIN_SIZE + params.HISTOGRAM.STARTING_UV;
  assert(max(abs(Y_(:) - Y(:))) <= (params.HISTOGRAM.BIN_SIZE * 0.5 + 1e-5));

end

% Test that values map to the same bin when shifted by the width of the
% histogram.
for iter = 1:100

  params.HISTOGRAM.NUM_BINS = ceil(rand*100);
  params.HISTOGRAM.BIN_SIZE = 0.5 + rand;
  params.HISTOGRAM.STARTING_UV = ...
    randn - params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE/2;

  Y = randn(2, 100)*10;
  Y_idx = UvToIdx(Y, params);

  histogram_width = params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE;
  k = round(randn*3);
  Y_idx_ = UvToIdx(Y + k * histogram_width, params);

  assert(all(Y_idx_(:) == Y_idx(:)))

end

fprintf('Tests Passed\n');
