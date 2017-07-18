function params = ChengSonyA57ThumbConstants(params)
% Project specific parameters.

% Inherit the thumbnail Cheng constants.
addpath('./projects/ChengThumb/') % Assumes the current path is /matlab/
params = ChengThumbConstants(params);

% The path where the data used for cross-validation and training is kept.
params.TRAINING.CROSSVALIDATION_DATA_FOLDER ...
  = '/mnt/gcam-raid/barron/cheng/preprocessed/ChengThumb/SonyA57/';

params.HISTOGRAM.STARTING_UV = -0.25;
