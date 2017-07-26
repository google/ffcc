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

function params = ChengThumbConstants(params)
% Project specific parameters.

% Inherit the normal Cheng constants.
addpath('./projects/Cheng/') % Assumes the current path is /matlab/
params = ChengConstants(params);

% The expected resolution of the sensor stats.
params.SENSOR.STATS_SIZE = [32, 48];
% The expected bit depth of the feature stats
params.SENSOR.STATS_BIT_DEPTH = 8;
% Setting the CCM to nan will cause the importing code to use whatever
% input CCM is correct.
params.SENSOR.CCM = nan;
% If true, assume that the input stats are linear intensity. Otherwise, assume
% that the input stats are in gamma-corrected sRGB space.
params.SENSOR.LINEAR_STATS = true;

params.HISTOGRAM.USE_2015_CHANNELS = false;
