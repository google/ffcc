function params = ChengCanon600DConstants(params)
% Project specific parameters.

% Inherit the normal Cheng constants.
addpath('./projects/Cheng/') % Assumes the current path is /matlab/
params = ChengConstants(params);

% The path where the data used for cross-validation and training is kept.
paths = DataPaths;
params.TRAINING.CROSSVALIDATION_DATA_FOLDER ...
  = fullfile(paths.cheng, 'preprocessed/Cheng/Canon600D/');

params.HISTOGRAM.STARTING_UV = -0.34375;
