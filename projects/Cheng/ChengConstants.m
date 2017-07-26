% Copyright 2017 Google Inc.
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
%     https://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function params = ChengConstants(params)
% Project specific parameters.

% The path where the data used for cross-validation and training is kept.
params.TRAINING.EXTRA_DATA_FOLDERS = { ...
};
params.TRAINING.TUNING_ERROR_FUNCTION = @(x)geomean([x.rgb_err.mean, ...
  x.rgb_err.median, x.rgb_err.tri, x.rgb_err.b25, x.rgb_err.w25]);
% If true, a black body per-bin prior gets learned. Set to
% true if the photometric properties of your input are consistent (ie, all
% images will come from the same camera) and set to false otherwise (ie,
% you would like to run on arbitrary images from the internet).
params.TRAINING.LEARN_BIAS = true;

% The number of bins in the histogram, which is also the size of the FFT
% convolutions that must be performed.
params.HISTOGRAM.NUM_BINS = 64;
% The spacing between bins in UV coordinates, which determines the resolution
% of the predicted UV white points.
params.HISTOGRAM.BIN_SIZE = 1/32;
% The UV coordinate of the first bin in the histogram. Here it's set such
% that the center of the histogram roughly corresponds to a gain of [2, 1, 2],
% which is standard for cameras where green is gained up by 2x.
params.HISTOGRAM.STARTING_UV = -0.3125;
% Whether or not to ignore zero-valued pixels when constructing histograms
% from training data. This is a good idea if the training data contains
% color charts or saturated pixels which have been masked out.
params.HISTOGRAM.MASK_ZERO_PIXELS = true;

% The expected resolution of the sensor stats.
params.SENSOR.STATS_SIZE = [256, 384];
% The expected bit depth of the feature stats, 12 bit 2.7mp images downsampled
% to 0.1mp images should give at least 4 extra bits of precision.
params.SENSOR.STATS_BIT_DEPTH = 16;
% Setting the CCM to nan will cause the importing code to use whatever
% input CCM is correct.
params.SENSOR.CCM = nan;
% If true, assume that the input stats are linear intensity. Otherwise, assume
% that the input stats are in gamma-corrected sRGB space.
params.SENSOR.LINEAR_STATS = true;

params.TRAINING.GENERATE_GEHLERSHI_DATA = false;

params.HISTOGRAM.USE_2015_CHANNELS = false;
