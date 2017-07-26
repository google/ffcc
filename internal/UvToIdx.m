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

function Y_idx = UvToIdx(Y, params)
% Maps from a 2xn matrix of UV values to their nearest corresponding
% (possibly-aliased) histogram index.

assert(size(Y,1) == 2);

wrap = @(x)(mod(x, params.HISTOGRAM.NUM_BINS)+1);
Y_idx = wrap( ...
  round((Y - params.HISTOGRAM.STARTING_UV) / params.HISTOGRAM.BIN_SIZE));
