% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function params = LoadProjectParams(project_name)
% Produces a parameter struct for a given project. The struct defined in
% PrivateConstants.m is passed to {project_name}Constants.m, and then onto
% {project_name}Hyperparametrs.m (unless no hyperparameters exist for this
% project yet, in which case we use the default values specified in
% DefaultHyperparameters.m

% This function might get called from the root /ffcc/ folder or by
% the /ffcc/scripts/ subfolder, so we need to handle both cases.
cur_dir = pwd;
if strcmp(cur_dir((find(cur_dir == '/', 1, 'last')+1):end), 'ffcc')
  projects_dir = './projects';
else
  projects_dir = '../projects';
end

addpath(fullfile(projects_dir, project_name))

addpath(fullfile(projects_dir, project_name))
func_constants = str2func([project_name, 'Constants']);

hyperparams_filename = ...
  fullfile(projects_dir, project_name, [project_name, 'Hyperparams.m']);
if exist(hyperparams_filename, 'file')
  func_hyperparams = str2func([project_name, 'Hyperparams']);
else
  fprintf('%s does not exist, using default hyperparameters\n', ...
    hyperparams_filename)
  addpath(projects_dir)
  func_hyperparams = str2func('DefaultHyperparams');
end

params = func_hyperparams(func_constants(PrivateConstants));

params.project_name = project_name;

% The location of the learned model, used by Train().
params.output_model_filename = ...
  fullfile(params.TRAINING.OUTPUT_MODEL_FOLDER, [params.project_name, '.mat']);

% The location of visualizations of the model, used by Visualize().
params.output_visualization_folder = ...
  fullfile(params.TRAINING.OUTPUT_VISUALIZATION_ROOT, params.project_name);
