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

function N = Psplat2(u, v, c, bin_lo, bin_step, n_bins)
% Splat a bunch of (u, v) coordinates to a 2D histogram. We splat with a
% periodic edge condition, which is important because we will later
% convolve with periodic edges (FFT)

ub = 1 + mod(round((u - bin_lo) / bin_step), n_bins);
vb = 1 + mod(round((v - bin_lo) / bin_step), n_bins);
N = reshape(accumarray(sub2ind([n_bins, n_bins], ub(:), vb(:)), c(:), ...
                               [n_bins^2, 1]), n_bins, n_bins);
