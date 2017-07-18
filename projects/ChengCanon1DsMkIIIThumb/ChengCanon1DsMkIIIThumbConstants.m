function params = ChengCanon1DsMkIIIThumbConstants(params)
% Project specific parameters.

% Inherit the thumbnail Cheng constants.
addpath('./projects/ChengThumb/') % Assumes the current path is /matlab/
params = ChengThumbConstants(params);

% The path where the data used for cross-validation and training is kept.
paths = DataPaths;
params.TRAINING.CROSSVALIDATION_DATA_FOLDER ...
  = fullfile(paths.cheng, 'preprocessed/ChengThumb/Canon1DsMkIII/');

params.HISTOGRAM.STARTING_UV = -0.3125;
