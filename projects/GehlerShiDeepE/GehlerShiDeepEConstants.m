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

function params = GehlerShiDeepConstants(params)
% Project specific parameters.

% Inherit the normal Gehler-Shi constants.
addpath('./projects/GehlerShi/') % Assumes the current path is /matlab/
params = GehlerShiConstants(params);

params.DEEP.ENABLED = true;
params.DEEP.FEATURE_FILENAME_TAGS = {'exif'};
params.DEEP.NUM_HIDDEN_UNITS = 4;
params.DEEP.WHITEN_FEATURES = true;
params.DEEP.PRECONDITION_WEIGHTS=true;
params.TRAINING.GENERATE_GEHLERSHI_DATA = false;
params.TRAINING.NUM_ITERS_ANNEAL = 1;
params.TRAINING.NUM_ITERS_LBFGS_FINAL = 64;
params.TRAINING.DISABLE_REGULARIZER = true;
