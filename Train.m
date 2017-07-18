function Train(project_name, additional_code)
% Given a project name, trains a model on all of the training data for that
% project, and writes the learned model (as a Matlab .mat file) to disk.
% "additional_code" is an optional string of Matlab code that is evaluated
% before doing this search, and so can be used to modify the project's default
% parameters for experimentation.

addpath(genpath('./internal/'))
params = LoadProjectParams(project_name);

if nargin < 2
  additional_code = '';
end

% Evaluate any additional code that might have been passed in.
assert(ischar(additional_code));
if ~isempty(additional_code)
  fprintf('Additional code: %s\n', additional_code)
  eval(additional_code)
end

% If we are doing cross-validation, then set the number of folds to 1.
if params.TRAINING.CROSSVALIDATION.NUM_FOLDS > 1
  params.TRAINING.CROSSVALIDATION.NUM_FOLDS = 1;
end

% Load all training data and as one cross-validation "fold".
data = PrecomputeMixedData(...
  params.TRAINING.CROSSVALIDATION_DATA_FOLDER, ...
  params.TRAINING.EXTRA_TRAINING_DATA_FOLDERS, ...
  {}, params);

% Train the cross-validation "models".
models = TrainModel(data, params);

% extract the first and only model.
assert(length(models) == 1)
model = models{1};

% Save the trained model.
fprintf('Saving model to %s\n', params.output_model_filename);
save(params.output_model_filename, 'model');
