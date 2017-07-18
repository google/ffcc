function params = ChengPanasonicGX1ThumbConstants(params)
% Project specific parameters.

% Inherit the thumbnail Cheng constants.
addpath('./projects/ChengThumb/') % Assumes the current path is /matlab/
params = ChengThumbConstants(params);

% The path where the data used for cross-validation and training is kept.
paths = DataPaths;
params.TRAINING.CROSSVALIDATION_DATA_FOLDER ...
  = fullfile(paths.cheng, 'preprocessed/ChengThumb/PanasonicGX1/');

params.HISTOGRAM.STARTING_UV = -0.375;
