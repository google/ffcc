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

function IdxToUV_Test

% Test that IdxToUv is consistent with UvToIdx.
for iter = 1:100

  params.HISTOGRAM.NUM_BINS = ceil(rand*100);
  params.HISTOGRAM.BIN_SIZE = 0.5 + rand;
  params.HISTOGRAM.STARTING_UV = ...
    randn - params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE/2;

  bins = EnumerateBins(params);

  Y_idx = (1 + rand(2, 100) * (params.HISTOGRAM.NUM_BINS-1));
  assert(all(Y_idx(:) >= 1))
  assert(all(Y_idx(:) <= params.HISTOGRAM.NUM_BINS))
  Y = IdxToUv(Y_idx, params);

  Y_idx_recon = UvToIdx(Y, params);
  assert(all(all(Y_idx_recon == round(Y_idx))));

  % Check that the mean + variance interface (3 arguments) to IdxToUv produces
  % identical output to the mean-only interface (2 arguments).
  [Y2, ~] = IdxToUv(Y_idx, nan, params);
  assert(all(Y2(:) == Y(:)))

  % Sanity-check the wrap-around behavior of the histograms by checking that
  % the index at the uv extent - epsilon is the last index, and the index at
  % the uv extent + epsilon is the first index.
  [bins, extents] = EnumerateBins(params);

  y1 = [extents(1) - 1e-10; extents(1) - 1e-10];
  idx1 = UvToIdx(y1, params);
  assert(all(idx1 == params.HISTOGRAM.NUM_BINS));

  y2 = [extents(1) + 1e-10; extents(1) + 1e-10];
  idx2 = UvToIdx(y2, params);
  assert(all(idx2 == 1));

end

fprintf('Tests Passed\n');
