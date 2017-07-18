function params = ChengCanon1DsMkIIIConstants(params)
% Project specific parameters.

% Inherit the normal Cheng constants.
addpath('./projects/Cheng/') % Assumes the current path is /matlab/
params = ChengConstants(params);

% The path where the data used for cross-validation and training is kept.
paths = DataPaths;
params.TRAINING.CROSSVALIDATION_DATA_FOLDER ...
  = fullfile(paths.cheng, 'preprocessed/Cheng/Canon1DsMkIII/');

params.HISTOGRAM.STARTING_UV = -0.3125;
