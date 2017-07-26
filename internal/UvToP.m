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

function P = UvToP(y, params)
% Maps from a 2x1 vector of UV values to a bilinearly interpolated PDF.

assert(size(y,1) == 2);
assert(size(y,2) == 1);

y_ij = (y - params.HISTOGRAM.STARTING_UV) / params.HISTOGRAM.BIN_SIZE;

y_lo = floor(y_ij);
y_hi = y_lo + 1;

w_1 = y_ij - y_lo;
w_0 = 1 - w_1;
w_00 = w_0(1) * w_0(2);
w_01 = w_0(1) * w_1(2);
w_10 = w_1(1) * w_0(2);
w_11 = w_1(1) * w_1(2);

wrap = @(x)(mod(x, params.HISTOGRAM.NUM_BINS)+1);

P = sparse( ...
  [wrap(y_lo(1)); wrap(y_lo(1)); wrap(y_hi(1)); wrap(y_hi(1))], ...
  [wrap(y_lo(2)); wrap(y_hi(2)); wrap(y_lo(2)); wrap(y_hi(2))], ...
  [w_00; w_01; w_10; w_11], ...
  params.HISTOGRAM.NUM_BINS, params.HISTOGRAM.NUM_BINS);
