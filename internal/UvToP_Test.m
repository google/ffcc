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

function UvToP_Test

for iter = 1:100

  params.HISTOGRAM.NUM_BINS = ceil(rand*100);
  params.HISTOGRAM.BIN_SIZE = 0.5 + rand;
  params.HISTOGRAM.STARTING_UV = ...
    randn - params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE/2;

  bins = EnumerateBins(params);

  y = randn(2, 1) * 10;
  y_idx = UvToIdx(y, params);

  P = UvToP(y, params);

  % Check that the argmax of P is the integer location returned by UvToIdx.
  assert(P(y_idx(1), y_idx(2)) >= max(P(:)));

  % Check that the expectation of P is equal to y, but only if y is within the
  % the histogram.
  if (all(y >= bins(1)) & all(y <= bins(end)))
    [uu, vv] = ndgrid(bins, bins);
    assert( abs(sum(uu(:) .* P(:)) - y(1)) < 1e-12 )
    assert( abs(sum(vv(:) .* P(:)) - y(2)) < 1e-12 )
  end

  % Test that P is the same when y is shifted by the width of the histogram.
  histogram_width = params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE;
  k = round(randn*3);
  P_ = UvToP(y + k * histogram_width, params);

  assert( max(abs(P_(:) - P(:))) < 1e-12 );

  % Sanity-check the wrap-around behavior of the histograms by checking that the
  % PDFs at the extents of the histogram are identical.
  [bins, extents] = EnumerateBins(params);

  y1 = [extents(1); extents(1)];
  P1 = UvToP(y1, params);

  y2 = [extents(2); extents(2)];
  P2 = UvToP(y2, params);

  assert(full(max(abs(P1(:) - P2(:)))) < 1e-10);

  y1 = [extents(2); extents(1)];
  P1 = UvToP(y1, params);

  y2 = [extents(1); extents(2)];
  P2 = UvToP(y2, params);

  assert(full(max(abs(P1(:) - P2(:)))) < 1e-10);

end

fprintf('Tests Passed\n');
