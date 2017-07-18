function Visualize(project_name, additional_code)
% Visualize a project by performing cross-validation and evaluating each image
% with its respective cross-validation model, while dumping the white-balanced
% output into the project's output_visualization_folder.
% TODO(barron): better integrate this with Test()

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

data = PrecomputeMixedData(...
  params.TRAINING.CROSSVALIDATION_DATA_FOLDER, ...
  params.TRAINING.EXTRA_TRAINING_DATA_FOLDERS, ...
  params.TRAINING.EXTRA_TESTING_DATA_FOLDERS, params);

CrossValidate(data, params, params.output_visualization_folder);
