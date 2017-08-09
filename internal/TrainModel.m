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

function [models, train_metadata, params] = TrainModel(data, params)
% Read in training data and parameters, return the calibrated new models.

addpath(genpath('./minFunc'));
addpath(genpath('./lib_fft'))
addpath(genpath('./lib_flatten'))

% Make sure all data(.).is_test is logical and of the same (1 x N_FOLDS) size.
assert(all(arrayfun(@(x) islogical(x.is_test), data)))
assert(all(arrayfun(@(x) size(x.is_test,1) == 1, data)))

if params.TRAINING.CROSSVALIDATION.NUM_FOLDS == 1
  fprintf(['WARNING: Cross-validation settings are invalid, ' ...
           'you are training on the test\nset and that is very bad ', ...
           'unless you are training a final model\n']);
end

% Check that all histograms sum to the expected value.
% Normalized histograms sum to one, although rarely histograms may sum to 0
% (if an image is all black and black pixels have been masked out during
% feature construction).
X_sums = arrayfun(@(x) sum(sum(x.X,2),1), data, 'UniformOutput', false);
X_sums = squeeze(cat(3, X_sums{:}));
assert(all((abs(X_sums - 1) < 1e-8) | (abs(X_sums) < 1e-8)))

rng('default')
models = {};
n_folds = length(data(1).is_test);
train_metadata.final_losses = nan(1, n_folds);
train_metadata.train_times = nan(1, n_folds);
train_metadata.opt_traces = ...
  repmat({[]}, params.TRAINING.NUM_ITERS_ANNEAL, n_folds);

for i_fold = 1:n_folds

  start_time = clock;

  % This global value is used as a timer to determine when to visualize the
  % current model during training, so we clear the value before optimization
  % to restart timer.
  global display_last_time
  display_last_time = [];

  % Grab the training data corresponding to the current fold.
  is_test = cat(1, data.is_test);
  keep = full(~is_test(:,i_fold));
  data_fold = data(keep);

  if params.DEEP.ENABLED && params.DEEP.WHITEN_FEATURES
    % Compute a whitening transformation from the feature vectors, and whiten
    % all of the feature vectors "in place". This requires that we unwhiten
    % the learned weights later.
    X = cat(2, data_fold.feature_vec);
    whitening_transformation = ComputeWhiteningTransformation( ...
      X, true, params.HYPERPARAMS.WHITEN_VARIANCE_PAD, 'zca');
    X_white = whitening_transformation.whiten(X);
    for i = 1:length(data_fold)
      data_fold(i).feature_vec = X_white(:, i);
    end
  end

  % Precompute the FFT for each datapoint's histogram.
  for i_data = 1:length(data_fold)
    data_fold(i_data).X_fft = fft2(double(data_fold(i_data).X));
  end

  % Precompute the weights used to regularize model.F_fft, which correspond to
  % a total variation measure in the Fourier domain.
  X_sz = size(data_fold(1).X);
  if numel(X_sz) == 2
    X_sz(3) = 1;
  end
  u_variation_fft = abs(fft2([-1; 1]/sqrt(8), X_sz(1), X_sz(2))).^2;
  v_variation_fft = abs(fft2([-1, 1]/sqrt(8), X_sz(1), X_sz(2))).^2;
  total_variation_fft = u_variation_fft + v_variation_fft;

  % A helper function for applying a scale and shift to a stack of images.
  apply_scale_shift = @(x, m, b)...
    bsxfun(@plus, ...
      bsxfun(@times, x, permute(m(:), [2,3,1])), permute(b(:), [2,3,1]));

  regularizer.F_fft = apply_scale_shift(total_variation_fft, ...
    params.HYPERPARAMS.FILTER_MULTIPLIERS, ...
    params.HYPERPARAMS.FILTER_SHIFTS);

  if params.TRAINING.LEARN_BIAS
    regularizer.B_fft = apply_scale_shift(total_variation_fft, ...
      params.HYPERPARAMS.BIAS_MULTIPLIER, ...
      params.HYPERPARAMS.BIAS_SHIFT);
  end

  % Construct the initial model.
  shallow_model = [];
  shallow_model.F_fft_latent = zeros(X_sz(1) * X_sz(2), X_sz(3));
  if params.TRAINING.LEARN_BIAS
    shallow_model.B_fft_latent = zeros(X_sz(1) * X_sz(2), 1);
  end
  [shallow_model_vec, params.shallow_model_meta] = ...
    BlobToVector(shallow_model);

  % Precache a mapping from real vectors to complex FFT coefficients.
  params.fft_mapping = Fft2ToVecPrecompute([X_sz(1), X_sz(2)]);

  preconditioner.F_fft_latent = ...
    Fft2RegularizerToPreconditioner(regularizer.F_fft);

  if params.TRAINING.LEARN_BIAS
    preconditioner.B_fft_latent = ...
      Fft2RegularizerToPreconditioner(regularizer.B_fft);
  end

  if params.DEEP.ENABLED
    input_length = length(data(1).feature_vec);
    output_length = length(shallow_model_vec);
    num_hidden = params.DEEP.NUM_HIDDEN_UNITS;
    rand_sigma = params.HYPERPARAMS.RANDOM_INIT_SIGMA;
    model = struct();
    model.W = {};
    model.b = {};
    if isempty(params.DEEP.NUM_HIDDEN_UNITS)
      % If no hidden units are specified then we make a "shallow" linear model.
      sz = [output_length, input_length];
      model.W{1} = rand_sigma*randn(sz(1), sz(2));
      model.b{1} = rand_sigma*randn(sz(1), 1);
    else
      for d = 1:(length(num_hidden)+1)
        if d == 1
          sz = [num_hidden(d), input_length];
        elseif d == (length(num_hidden)+1)
          sz = [output_length, num_hidden(d-1)];
        else
          sz = [num_hidden(d), num_hidden(d-1)];
        end
        model.W{d} = rand_sigma*randn(sz(1), sz(2));
        model.b{d} = rand_sigma*randn(sz(1), 1);
      end
    end
  else
    model = shallow_model;
  end

  % Collapse the model struct down to a vector, while preserving the metadata
  % necessary to reconstruct it.
  [model_vec, params.model_meta] = BlobToVector(model);

  if params.DEEP.ENABLED
    lossfun = @TrainModelLossfunDeep;
  else
    lossfun = @TrainModelLossfun;
  end

  % To train our model we do several rounds NUM_ITERS_ANNEAL of LBFGS,
  % where the loss at each annealing iteration is a linear interpolation
  % between the convex cross-entropy loss and the non-convex Von Mises
  % loss.
  for i_anneal = 1:params.TRAINING.NUM_ITERS_ANNEAL
    if params.DEBUG.DISPLAY_LBFGS
      fprintf('Fold %d  //  Anneal %d\n', i_fold, i_anneal);
    end

    % The loss multipliers are linear combinations such that the first
    % annealing pass solely minimizes the convex loss, and the last
    % annealing pass solely minimizes the non-convex loss.
    if params.TRAINING.NUM_ITERS_ANNEAL == 1
      params.loss_mult.crossent = params.HYPERPARAMS.CROSSENT_MULTIPLIER;
      params.loss_mult.vonmises = params.HYPERPARAMS.VONMISES_MULTIPLIER;
    else
      vonmises_weight = (i_anneal - 1) / (params.TRAINING.NUM_ITERS_ANNEAL-1);
      params.loss_mult.crossent = (1 - vonmises_weight) * ...
        params.HYPERPARAMS.CROSSENT_MULTIPLIER;
      if isinf(params.HYPERPARAMS.VONMISES_MULTIPLIER)
        params.loss_mult.vonmises = vonmises_weight;
      else
        params.loss_mult.vonmises = vonmises_weight * ...
          params.HYPERPARAMS.VONMISES_MULTIPLIER;
      end
    end

    if params.DEBUG.DISPLAY_LBFGS
      fprintf('Crossent Mult = %f, Von Mises Mult = %f\n', ...
        params.loss_mult.crossent, params.loss_mult.vonmises);
    end

    % The number of iterations for pass of LBFGS is produced by interpolating
    % two constants in log-space, which allows us to control the relative
    % number of iterations spent minimizing the convex or non-convex loss.
    if params.TRAINING.NUM_ITERS_ANNEAL == 1
      num_iters = params.TRAINING.NUM_ITERS_LBFGS_FINAL;
    else
      iter_weight = (i_anneal - 1) / (params.TRAINING.NUM_ITERS_ANNEAL - 1);
      num_iters = round(exp(...
        log(params.TRAINING.NUM_ITERS_LBFGS_INITIAL) * (1-iter_weight) + ...
        log(params.TRAINING.NUM_ITERS_LBFGS_FINAL) * iter_weight));
    end

    % Check the correctness of the loss function's analytical gradient.
    if params.DEBUG.CORRECT_GRADIENT
      data_sub = data_fold(1:ceil(length(data_fold)/3):end);
      CheckGradient(lossfun, model, 1e-5, 0.001, 20, 500, ...
        {data_sub, regularizer, preconditioner, params});
    end

    % Test "how convex" the loss function is.
    if params.DEBUG.CONVEX_LOSS
      CheckConvex(lossfun, model, data, 80*3, 3, ...
        {regularizer, preconditioner, params});
    end

    % Test that the loss function is additive.
    if params.DEBUG.ADDITIVE_LOSS
      CheckAdditive(lossfun, model, data_fold, 8, ...
        {regularizer, preconditioner, params});
    end

    lbfgs_options = struct( ...
      'Method', 'lbfgs', ...
      'MaxIter', num_iters, ...
      'Corr', num_iters, ...
      'MaxFunEvals', 4 + 2*num_iters, ...
      'optTol', 0, ...
      'progTol', 0);
    if params.DEBUG.DISPLAY_LBFGS
      lbfgs_options.Display = 'iter';
    else
      lbfgs_options.Display = 'off';
    end

    [model_vec, ~, ~, output] = minFunc(lossfun, model_vec, lbfgs_options, ...
      data_fold, regularizer, preconditioner, params);

    train_metadata.opt_traces{i_anneal, i_fold} = output.trace.fval(:)';

    model = VectorToBlob(model_vec, params.model_meta);
  end

  if params.TRAINING.TIMES_ARE_INVALID
    train_metadata.train_times(i_fold) = nan;
  else
    train_metadata.train_times(i_fold) = etime(clock, start_time);
  end

  if ~params.DEEP.ENABLED
    % Compute the loss of the final model, for use in evaluating different
    % optimization techniques. The loss function also returns the non-vectorized
    % and rescaled model parameters as a side effect, so we use that
    % functionality to produce the output model.
    [train_metadata.final_losses(i_fold), ~, model] = TrainModelLossfun( ...
      model_vec, data_fold, regularizer, preconditioner, params);

    model = rmfield(model, 'F_fft_latent');

    if params.TRAINING.LEARN_BIAS
      % Remove unnecessary black-body metadata from the model.
      model = rmfield(model, ...
        {'B_fft_latent', 'B_fft'});
    else
      model.B = zeros(X_sz(1), X_sz(2));
    end

    models{i_fold} = model;

  else

    train_metadata.final_losses(i_fold) = TrainModelLossfunDeep( ...
      model_vec, data_fold, regularizer, preconditioner, params);

    model = VectorToBlob(model_vec, params.model_meta);

    if params.DEEP.WHITEN_FEATURES
      % Unwhiten the first layer according to the whitening transformation, so
      % that is produces the correct output on the unwhitened feature vectors.
      model.W{1} = model.W{1} * whitening_transformation.A;
      model.b{1} = model.W{1} * whitening_transformation.b + model.b{1};
    end
    models{i_fold} = model;

  end
end

params.preconditioner = preconditioner;
params.regularizer = regularizer;
