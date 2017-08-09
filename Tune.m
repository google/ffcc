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

function Tune(project_name, overwrite_parameters, additional_code)
% Tune(project_name, overwrite_parameters, additional_code)
% Tunes the hyperparameters for a project. Given a project name, takes the
% current parameter settings that need to be tuned under cross-validation and
% performs crude coordinate descent of those parameters, one parameter at a
% time, while minimizing some cross-validation loss.
% Inputs:
%   project_name - The name of a project being tuned, which must correspond to
%     a folder in ./projects/.
%   overwrite_parameters - If true, open the current best Hyperparameters
%     file for the project (./{project_name}/{project_name}Hyperparams.m)
%     and dump the tuned hyperparameters directly into that file. If false,
%     dump the output into a new file in the project folder with a
%     timestamp's filename, which can be manually copied into the main
%     *Hyperparams.m file by the user. By default this option is set to true.
%   additional_code - an optional string of Matlab code that is evaluated
%     before doing this search, and so can be used to modify the project's
%     default parameters for experimentation. This code is included in the
%     output hyperparameter file as a comment. Because this code can include
%     matlab comments, this parameter can also be used to add comments to the
%     output file to make experiments easier to parse.

addpath(genpath('./internal/'))

if nargin < 2
  overwrite_parameters = true;
end

assert(islogical(overwrite_parameters));

if overwrite_parameters
  filename = fullfile('projects', project_name, ...
    [project_name, 'Hyperparams.m']);
else
  filename = fullfile('projects', project_name, [TimeString, '_Hyperparams.m']);
end
fprintf('Tuning Filename = %s\n', filename);

params = LoadProjectParams(project_name);

if nargin < 3
  additional_code = '';
end

% Evaluate any additional code that might have been passed in.
assert(ischar(additional_code));
if ~isempty(additional_code)
  fprintf('Additional code: %s\n', additional_code)
  eval(additional_code)
end

error_function = params.TRAINING.TUNING_ERROR_FUNCTION;

% This will be printed at the head of the resulting output file.
preamble = sprintf(['function [params, metrics] = %sHyperparams(params)\n' ...
  '%% The hyperparameters for this project, produced using Tune(). See\n', ...
  '%% ../DefaultHyperparams.m for documentation.\n', ...
  '%% Tuning started at %s.\n'], project_name, ...
  datestr(now, 'yyyy-mm-dd, HH:MM:SS:FFF'));
if ~isempty(additional_code)
  % Adding the additional code to the preamble as a comment. This code is
  % not meant to be evaluated in the hyperparameters file, this is just
  % meant to make it easier to review past experiments.
  preamble = [preamble, sprintf('%% Additional Code: %s\n', additional_code)];
end

hyperparams = params.HYPERPARAMS;
params = rmfield(params, 'HYPERPARAMS');

% If no bias is to be learned, then the parameters governing its
% regularization and preconditioning need not be tuned.
if ~params.TRAINING.LEARN_BIAS
  if isfield(hyperparams, 'BIAS_REGULARIZER')
    hyperparams = rmfield(hyperparams, 'BIAS_REGULARIZER');
  end
end

data = PrecomputeMixedData(...
  params.TRAINING.CROSSVALIDATION_DATA_FOLDER, ...
  params.TRAINING.EXTRA_TRAINING_DATA_FOLDERS, ...
  params.TRAINING.EXTRA_TESTING_DATA_FOLDERS, params);

TuneParameters(filename, params, data, error_function, hyperparams, ...
  params.TUNING, preamble);
