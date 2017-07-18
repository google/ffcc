function params = ChengSonyA57Constants(params)
% Project specific parameters.

% Inherit the normal Cheng constants.
addpath('./projects/Cheng/') % Assumes the current path is /matlab/
params = ChengConstants(params);

% The path where the data used for cross-validation and training is kept.
params.TRAINING.CROSSVALIDATION_DATA_FOLDER ...
  = '/mnt/gcam-raid/barron/cheng/preprocessed/Cheng/SonyA57/';

params.HISTOGRAM.STARTING_UV = -0.25;
