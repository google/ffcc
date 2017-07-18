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

function [params, history] = TuneParameters(filename, params, data, ...
  error_function, hyperparams, tune_params, preamble)
% Performs hacky coordinate descent over a set of parameters, by basically
% iterating over each parameter and multiplying and dividing its current value
% by some constant, and taking the new value if it improves some error metric.
% Inputs:
%   filename - The filename that the output of tuning will be written to.
%   params - The input parameter settings which are being tuned.
%   data - An array of datapoints, in the format produced by Precompute*Data(),
%     with which we will perform cross-validation.
%   error_function - A function handle which evaluates the error metric struct
%     produced by CrossValidate() and produces a scalar, which is what this
%     coordinate descent will greedily minimize.
%   hyperparams - The struct of hyperparameters that we will tune.
%   tune_params - A struct of parameters that determine how coordinate descent
%     should behave, see Tune() for an example.
%   preamble - A string which will be written to the beginning of the output
%     file.
% Outputs:
%   params - The optimally tuned parameters.
%   history - A recording of all explored hyperparmeter coordinates and their
%     losses.

addpath(genpath('./lib_flatten'))

% The error of the current parameter estimate, inf means that no parameter
% settings have been evaluated yet.
min_error = inf;

[hyperparams_vec, hyperparams_meta] = BlobToVector(hyperparams);

if isfield(params.TUNING, 'ONLY_TUNE_FIELDS') ...
    && ~isempty(params.TUNING.ONLY_TUNE_FIELDS)
  hyperparams_do_search = VectorToBlob( ...
    false(size(hyperparams_vec)), hyperparams_meta);
  for i = 1:length(params.TUNING.ONLY_TUNE_FIELDS)
    hyperparams_do_search.(params.TUNING.ONLY_TUNE_FIELDS{i})(:) = true;
  end
  hyperparams_do_search = BlobToVector(hyperparams_do_search);
else
  hyperparams_do_search = true(size(hyperparams_vec));
end

first_evaluation = true;

history.hyperparams = [];
history.errors  = [];

% The current multiplier value for each hyperparameter being searched over.
tune_mults = tune_params.INIT_MULT * hyperparams_do_search;

% We will perform no more than NUM_ITERS passes over all variables for each
% tuning power.
for i_pass = 1:tune_params.NUM_ITERS;

  % Iterate over each variable name.
  for i_var = find(hyperparams_do_search)'

    % Grab the variable name and its current value.
    cur_val = hyperparams_vec(i_var);

    % Hyperparameters which are set to 0 cannot be rescaled, so we can save
    % time by skipping them.
    if isinf(cur_val) || (cur_val == 0)
      if tune_params.DISPLAY
        fprintf('Var %d = 0 or infinity, skipping this variable\n', i_var);
      end
      continue;
    end

    % Determine what values to try to the current variable.
    tune_mult = max(tune_params.MIN_MULT, tune_mults(i_var));
    if first_evaluation
      % On the first iteration, we would like a baseline estimate of the error
      % with the input parameters, so we set the first multiplier to try to 1.
      try_mults = [1, 1/tune_mult, tune_mult];
      first_evaluation = false;
    else
      try_mults = [1/tune_mult, tune_mult];
    end

    winner_found = false;
    negative_improvement = false;
    for i_mult = 1:length(try_mults)

      % Construct a new params struct where the current variable is set to the
      % new value being tried.
      new_hyperparams_vec = hyperparams_vec;
      new_hyperparams_vec(i_var) = cur_val .* try_mults(i_mult);

      if (log(new_hyperparams_vec(i_var)) > (log(cur_val) + 1e-5)) ...
          && negative_improvement
        if tune_params.DISPLAY
          fprintf('Optimal multiplier is < 2^%g, skipping\n', log2(cur_val));
        end
        continue;
      end

      new_params = params;
      new_params.HYPERPARAMS = ...
        VectorToBlob(new_hyperparams_vec, hyperparams_meta);

      if (i_pass == 1) && (i_var == 1) && (i_mult == 1)
        if tune_params.DISPLAY
          fprintf('Evaluating default parameters\n');
        end
      else
        if tune_params.DISPLAY
          fprintf('Trying Var %d = 2^(%g%+g) = 2^%g\n', i_var, ...
            log2(cur_val), log2(try_mults(i_mult)), ...
            log2(new_hyperparams_vec(i_var)));
        end
      end

      if ~isempty(history.hyperparams)
        valid = all(history.hyperparams ~= 0, 2);
        X = log(history.hyperparams(valid,:));
        x = log(new_hyperparams_vec(valid));
        diff = bsxfun(@minus, x, X);
        if any(max(abs(diff), [], 1) < 1e-3)
          if tune_params.DISPLAY
            fprintf('These hyperparams have been tried before, skipping\n');
          end
          continue;
        end
      end

      % Evaluate the proposed error metrics by performing cross-validation.
      metrics = CrossValidate(data, new_params);

      % Reduce the cross-validation statistics down to some error value.
      new_error = error_function(metrics);

      history.hyperparams = [history.hyperparams, new_hyperparams_vec];
      history.errors = [history.errors, new_error];

      % If the error produced by the new parameters is less than the minimum
      % error seen so far, take it.
      if new_error < min_error
        if (i_pass == 1) && (i_var == 1) && (i_mult == 1)
          if tune_params.DISPLAY
            fprintf('Baseline Error: %f\n', new_error);
          end
        else
          winner_found = true;
          if tune_params.DISPLAY
            fprintf('%f < %f, accepting this new value.\n', ...
              new_error, min_error);
          end
        end

        if log(new_hyperparams_vec(i_var)) < (log(cur_val)-1e-5)
          negative_improvement = true;
        end

        min_metrics = metrics;
        min_error = new_error;
        hyperparams_vec = new_hyperparams_vec;
        params = new_params;

        % Print the new parameters, in case the user wants to interrupt.
        if tune_params.DISPLAY
          fprintf('Current Best Parameters:\n');
        end

        if tune_params.DISPLAY
          fprintf('params.HYPERPARAMS = %s;\n', BlobToString( ...
            params.HYPERPARAMS, true));
        end

        if ~isempty(filename)
          fid = fopen(filename, 'w');
          fprintf(fid, '%s\n', preamble);
          fprintf(fid, 'params.HYPERPARAMS = %s;\n', BlobToString( ...
            params.HYPERPARAMS, true));
          fprintf(fid, '\n');
          fprintf(fid, 'metrics = %s;\n', BlobToString(metrics));
          fprintf(fid, '\n');
          fprintf(fid, '%% Tuning error = %f\n', min_error);
          fprintf(fid, ...
            '%% i_pass = %d, i_var = %d, i_mult = %d\n', ...
            i_pass, i_var, i_mult);
          fprintf(fid, '%% Hyperparams written at %s\n', ...
            datestr(now, 'yyyy-mm-dd, HH:MM:SS:FFF'));
          fclose(fid);
        end

        if isfield(params.TRAINING, 'DO_NOT_TUNE') ...
            && params.TRAINING.DO_NOT_TUNE
          fprintf('Project flagged as DO_NOT_TUNE, returning\n');
          return;
        end

      else
        if tune_params.DISPLAY
          fprintf('%f >= %f, trying something else.\n', new_error, min_error);
        end
      end
    end

    if winner_found
      tune_mults(i_var) = tune_mults(i_var)^tune_params.MULT_INCREASE;
    else
      tune_mults(i_var) = tune_mults(i_var)^tune_params.MULT_DECREASE;
    end

  end

  if (all(tune_mults < (tune_params.MIN_MULT - 1e-8)))
    fprintf('All multipliers below %f, terminating\n', tune_params.MIN_MULT);
    break
  end

end

if tune_params.DISPLAY
  fprintf('Error: %f\n\n', min_error)
  fprintf('Current Metrics:\n');
  fprintf('metrics = %s;\n\n', BlobToString(min_metrics));
  fprintf('Current Hyperparams:\n');
  fprintf('params.HYPERPARAMS = %s;\n', BlobToString(params.HYPERPARAMS, true));
  fprintf('\n');
end
