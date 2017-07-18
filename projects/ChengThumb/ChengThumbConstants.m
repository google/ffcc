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
