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
