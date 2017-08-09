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

function [metrics, models] = CrossValidate(data, params, output_folder)
% CrossValidate -- Performs crossvalidation.
%
% Arguments:
%   data - an array of structs of datapoints, of the form returned by
%     PrecomputeTrainingData() or PrecomputeMixedData().
%   params - project specific tuning struct under projects/ folder.
%   output_folder - the path of a folder where the output of cross-validation
%     will be dumped (optional).
%
% Returns: the error metrics from cross validation, and a cell array of the
% trained models. The error metrics are surface statistics (mean, median, etc)
% computed on sets of underlying error measures (angle, negative log-
% likelihood). The measures we use are:
%   rgb_err - The angle in degrees between the estimated white point and
%     the true white point, in linear RGB space. Errors less than 2 degrees are
%     usually not perceptible.
%   uv_err - The Euclidian distance between the estimated white point and
%     the true white point, in our UV chroma space (U = log(R/G), V = log(B/G),
%     RGB are linear). This error is reported in units of histogram bins,
%     so this value gives an intuitive sense of how well-localized the white
%     point is in the actual space where localization happens.
%   vonmises_nll - The negative log-likelihood of the true white point under
%     a fitted Von Mises distribution in UV chroma space. This is similar to
%     uv_err, except that this takes into account the covariance matrix of the
%     white point, instead of just the location of the white point.
% The rgb_err statistics are commonly used in color constancy literature,
% but the vonmises_nll are more indicative of how well the AWB model will
% perform within a larger system, such as one that combines information or
% smooths white point estimates over time. See GetError.m for clarification
% of the surface statistics that are used.
%
% Usage example (see TuneParameters() for details):
%   data = PrecomputeTrainingData('/foo/train_folder', params);
%   CrossValidate(data, params);

addpath(genpath('./lib_flatten'))

assert(nargin >= 2);

% This flag is used to mock a loss function with a simple analytical form,
% for use in debugging TuneParameters.m.
if isfield(params.TUNING, 'DEBUG_MODE') && params.TUNING.DEBUG_MODE
  metrics.rgb_err.mean = sum(abs(log(BlobToVector(params.HYPERPARAMS))));
  return
end

if nargin < 3
  output_folder = [];
end

if ~isempty(output_folder)
  warning off all;
  mkdir(output_folder);
  warning on all;
end

% Train N_FOLDS models using cross-validation.
[models, train_metadata, params_trained] = TrainModel(data, params);

% Evaluate the error of each datapoint using the cross-validation model for
% which that datapoint is in the test-set.
rgb_outputs = nan(3, length(data));
rgb_gts = nan(3, length(data));
rgb_errs = nan(length(data), 1);
uv_errs = nan(2, length(data));
nlls = nan(length(data), 1);
filter_times = nan(length(data), 1);
is_training = false(length(data), 1);
for i_data = 1:length(data)

  if (params.TRAINING.CROSSVALIDATION.NUM_FOLDS <= 1)
    model_idx = 1;
  else
    model_idx = find(data(i_data).is_test, 1, 'first');
    assert(~isempty(model_idx))
  end
  model = models{model_idx};

  is_training(i_data) = ~any(data(i_data).is_test);

  % Estimate the white point and the loss of the ground-truth.
  X = data(i_data).X;
  Y = data(i_data).Y;
  avg_rgb = data(i_data).avg_rgb;

  if params.DEEP.ENABLED
    % TODO(barron): this is inelegant, find a cleaner way.
    [~, ~, output] = TrainModelLossfunDeep( ...
      BlobToVector(model), data(i_data), params_trained.regularizer, ...
      params_trained.preconditioner, params_trained);
    state_obs = output{1};
    metadata = output{2};
    losses = output{4};

    if ~params.TRAINING.TIMES_ARE_INVALID
      % Run the subset of the evaluation code just necesssary for white point
      % prediction, for profiling.
      start_time = clock;
      [~] = TrainModelLossfunDeep( ...
        BlobToVector(model), data(i_data), params_trained.regularizer, ...
        params_trained.preconditioner, params_trained);
      filter_times(i_data) = etime(clock, start_time);
    end
  else
    % Using the non-convex loss causes EvaluateModel to produce Von Mises
    % negative log-likelihoods, which is what we are trying to measure here.
    [state_obs, metadata, ~, losses] = EvaluateModel( ...
      model.F_fft, model.B, X, fft2(X), Y, [], avg_rgb, params);

    if ~params.TRAINING.TIMES_ARE_INVALID
      % Run the subset of the evaluation code just necesssary for white point
      % prediction, for profiling.
      start_time = clock;
      [~] = EvaluateModel( ...
        model.F_fft, model.B, X, fft2(X), [], [], avg_rgb, params);
      filter_times(i_data) = etime(clock, start_time);
    end
  end

  nlls(i_data) = losses.vonmises;

  % Compute the angle between the estimated illuminant and the true illuminant.
  uv_est = state_obs.mu;
  uv_true = Y;

  rgb_outputs(:, i_data) = UvToRgb(uv_est);
  rgb_gts(:, i_data) = UvToRgb(uv_true);
  rgb_err = (180 / pi) * acos(max(0, min(1, ...
    sum(rgb_outputs(:, i_data) .* rgb_gts(:, i_data)))));
  rgb_errs(i_data) = rgb_err;

  uv_errs(:, i_data) = (uv_est - uv_true) / params.HISTOGRAM.BIN_SIZE;

  if ~isempty(output_folder)

    mu = state_obs.mu;
    Sigma = state_obs.Sigma;

    if params.SENSOR.LINEAR_STATS
      CCM = data(i_data).CCM;
      if any(isnan(CCM(:)))
        CCM = params.SENSOR.CCM;
      end

      if ~isempty(data(i_data).original_filename) ...
          && exist(data(i_data).original_filename)
        I_linear = ReadImage(data(i_data).original_filename);
      else
        I_linear = imread(data(i_data).filename);
        I_linear = double(I_linear);
      end
    else
      CCM = eye(3);
      I_srgb = imread(data(i_data).filename);
      I_srgb = double(I_srgb) / 255;
      I_linear = UndoSrgbGamma(I_srgb);
    end

    if size(I_linear,1) > size(I_linear,2)
      I_linear = rot90(I_linear);
    end

    upsample_factor = sqrt(384*256) / sqrt(prod(params.SENSOR.STATS_SIZE));
    vis_size = round(params.SENSOR.STATS_SIZE * upsample_factor);

    % Use nearest-neighbor for upsampling, bilinear for downsampling.
    if size(I_linear,1) < vis_size(1)
      I_linear = imresize(I_linear, vis_size, 'nearest');
    else
      I_linear = imresize(I_linear, vis_size, 'bilinear');
    end
    I_linear = I_linear ./ max(I_linear(:));

    % Convert the white point into a RGB gain.
    gains_ours = [exp(mu(1)); 1; exp(mu(2))];
    gains_ours = gains_ours ./ min(gains_ours(:));

    % Compute the ground truth RGB gain.
    gains_true = [exp(Y(1)); 1; exp(Y(2))];
    gains_true = gains_true ./ min(gains_true(:));

    % Apply our gain and the true gain and the CCM.
    I_ours = min(1, bsxfun(@times, double(I_linear), ...
      permute(gains_ours(:), [2,3,1])));
    I_true = min(1, bsxfun(@times, double(I_linear), ...
      permute(gains_true(:), [2,3,1])));

    % Apply the CCM.
    I_input = min(1, reshape(reshape(I_linear, [], 3) * CCM', size(I_linear)));
    I_ours = min(1, reshape(reshape(I_ours, [], 3) * CCM', size(I_ours)));
    I_true = min(1, reshape(reshape(I_true, [], 3) * CCM', size(I_true)));

    % Normalize.
    I_input = I_input ./ max(I_input(:));
    I_true = I_true ./ max(I_true(:));
    I_ours = I_ours ./ max(I_ours(:));

    % Clip and tone map the white balanced image and the input image.
    I_input = uint8(255 * ApplySrgbGamma(max(0, min(1, I_input))));
    I_true = uint8(255 * ApplySrgbGamma(max(0, min(1, I_true))));
    I_ours = uint8(255 * ApplySrgbGamma(max(0, min(1, I_ours))));

    L_vis = CCM * UvToRgb(mu);
    L_vis = L_vis / max(L_vis(:));
    L_vis = uint8(255 * ApplySrgbGamma(max(0, min(1, L_vis))));
    L_vis = permute(L_vis, [2,3,1]);
    L_vis = imresize(L_vis, [256, 32]);

    Lt_vis = CCM * UvToRgb(uv_true);
    Lt_vis = Lt_vis / max(Lt_vis(:));
    Lt_vis = uint8(255 * ApplySrgbGamma(max(0, min(1, Lt_vis))));
    Lt_vis = permute(Lt_vis, [2,3,1]);
    Lt_vis = imresize(Lt_vis, [256, 32]);

    P_bvm = double( ...
      RenderHistogramGaussian(mu, Sigma, Y, size(I_ours,1), false, params))/255;
    linearize = @(x)(ApplySrgbGamma(max(0, min(1, ...
      reshape(reshape(x, [], 3) * CCM', size(x))))));
    P_bvm = uint8(round(linearize(P_bvm) * 255));
    vis = [I_true, I_ours, P_bvm];

    imwrite(vis, fullfile(output_folder, ...
      ['montage_', num2str(i_data, '%08d'), '.jpg']), 'Quality', 90);

    if params.TRAINING.DUMP_EXHAUSTIVE_VISUALIZATION
      imwrite(I_input, fullfile(output_folder, ...
        [num2str(i_data, '%08d'), '_input.jpg']), 'Quality', 90);

      imwrite(I_true, fullfile(output_folder, ...
        [num2str(i_data, '%08d'), '_true.jpg']), 'Quality', 90);

      imwrite(I_ours, fullfile(output_folder, ...
        [num2str(i_data, '%08d'), '_prediction.jpg']), 'Quality', 90);

      imwrite(L_vis, fullfile(output_folder, ...
        [num2str(i_data, '%08d'), '_illum.png']));

      imwrite(Lt_vis, fullfile(output_folder, ...
        [num2str(i_data, '%08d'), '_illum_true.png']));

      imwrite(P_bvm, fullfile(output_folder, ...
        [num2str(i_data, '%08d'), '_chroma.png']));

      P = single(metadata.P);
      save(fullfile(output_folder, [num2str(i_data, '%08d'), '_P.mat']), 'P');

      fid = fopen( ...
        fullfile(output_folder, [num2str(i_data, '%08d'), '_error.txt']), 'w');
      fprintf(fid, '%f', rgb_err);
      fclose(fid);

      fid = fopen(fullfile( ...
        output_folder, [num2str(i_data, '%08d'), '_confidence.txt']), 'w');
      fprintf(fid, '%f', metadata.entropy_confidence);
      fclose(fid);
    end
  end
end

% If a filename is specified, dump the estimated RGB values to it, where each
% image is reported as:
% filename R G B
if ~isempty(params.DEBUG.OUTPUT_SUMMARY)
  fid = fopen(params.DEBUG.OUTPUT_SUMMARY, 'w');
  for i_data = 1:length(data)
    filename = data(i_data).original_filename;
    filename = filename((1+find(filename == '/', 1, 'last')):end);
    fprintf(fid, '%s', filename);
    fprintf(fid, ' %f', rgb_outputs(:,i_data)');
    fprintf(fid, '\n');
  end
  fclose(fid);
end

assert(~any(isnan(rgb_errs(:))));
assert(~any(isnan(uv_errs(:))));
assert(~any(isnan(is_training(:))));
assert(~any(isnan(nlls(:))));

is_test = ~is_training;

% Compute the surface statistics for each error measure.
metrics.rgb_err = ErrorMetrics(rgb_errs(is_test));
metrics.uv_err = ErrorMetrics(sqrt(sum(uv_errs(:,is_test).^2,1)));
metrics.vonmises_nll = ErrorMetrics(nlls(is_test));
metrics.uv_bin_bias = mean(uv_errs(:, is_test),2)';

if any(is_training)
  metrics.training_rgb_err = ErrorMetrics(rgb_errs(is_training));
  metrics.training_uv_err = ...
    ErrorMetrics(sqrt(sum(uv_errs(:,is_training).^2,1)));
  metrics.training_vonmises_nll = ErrorMetrics(nlls(is_training));
  metrics.training_uv_bin_bias = mean(uv_errs(:, is_training),2)';
end

feature_times = cat(1, data.feature_time);
metrics.final_losses = train_metadata.final_losses;
metrics.train_times = train_metadata.train_times ...
  + sum(feature_times) / length(train_metadata.train_times);
metrics.min_feature_time = min(feature_times);
metrics.min_filter_time = min(filter_times);
metrics.median_feature_time = median(feature_times);
metrics.median_filter_time = median(filter_times);
metrics.opt_traces = train_metadata.opt_traces;
