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

function [bins, extents] = EnumerateBins(params)
% Takes as input a parameter struct, and produces a 1xNUM_BINS vector of the
% u/v value of each bin in the histogram implied by, STARTING_UV, NUM_BINS and
% BIN_SIZE.
% Also produces as optional output the extents of the histogram in UV
% coordinates (ie, the exact UV values at which the histogram "wraps around").

bins = params.HISTOGRAM.STARTING_UV + [0 : params.HISTOGRAM.BIN_SIZE ...
  : (params.HISTOGRAM.BIN_SIZE * (params.HISTOGRAM.NUM_BINS-1))];

extents = [bins(1) - params.HISTOGRAM.BIN_SIZE/2, ...
           bins(end) + params.HISTOGRAM.BIN_SIZE/2];
