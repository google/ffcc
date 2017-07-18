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

function [state_obs, meta, sub_loss, sub_losses, state_next] ...
  = EvaluateModel(F_fft, B, X, X_fft, Y, state_last, avg_rgb, params)
% Evaluates a an AWB model, and produces an estimate of the white point. If
% provided with the necessary input, this function can also be used to:
%   - Estimate a loss and gradient with respect to a ground-truth white point.
%   - Update a running estimate of a white point.
%   - Produce an estimate of the quality of the white point.
%
% Inputs:
%   F_fft  - A learned model's n-channel filter bank in frequency space.
%   B      - A learned model's per-chroma bias map. This argument is optional,
%            and if it is omitted then the bias is assumed to be all zeros.
%   X      - An n-channel feature histogram.
%   X_fft  - The n-channel FFT of X, which we take as input to avoid having
%            to repeatedly recompute the FFT of X during training.
%   Y      - The ground-truth (u, v) coordinate of the true white point. Set to
%            [] if the ground-truth is not known, or if this model is not being
%            used for training purposes.
%   state_last - A running model of the white point in a Kalman filter
%            framework, which is a struct with a mean "mu" and covariance
%            matrix "Sigma". Use [] as input if AWB is being run independently
%            on a set of independent images or if you don't care to update the
%            current state.
%   params - A struct of parameters.
%
% Outputs:
%   state_obs - A struct containing the estimated white point given just the
%            current input X, where "mu" and "Sigma" are a bivariate Von Mises
%            distribution modeling the estimated white point just using the
%            current input X.
%   meta   - Some metadata used to compute state_obs, useful for training
%            and visualization.
%   sub_loss - If Y is non-empty and params.loss_mult is defined, a struct
%            containing the weighted loss to be minimized and its gradient.
%   sub_losses - If Y is non-empty, this is a struct contaning the two
%            losses (cross-entropy and von-mises) that are used to create the
%            weighted average loss returned in "sub_loss". These are useful for
%            reporting errors.
%   state_next - A struct containing a bivariate Von Mises computed from the
%            current observation and the running estimate. This struct should
%            be used as input to the next frame.

assert(~isempty(X));
assert(~isempty(X_fft));

if isempty(B)
  B = zeros(size(X_fft,1), size(X_fft,2));
end

compute_gradient = (nargout >= 2);

% The filtered histogram H = sum(conv(F, X),3) + B. The convolution is
% performed with FFTs with the sum over channels performed in FFT-space so that
% ifft2() need only be called once, and some intermediate quantities are
% held onto as they are needed during backpropagation.
FX_fft = sum(bsxfun(@times, X_fft, F_fft), 3);
FX = real(ifft2(FX_fft));
H = FX + B;

% Turn the score histogram into a PDF by passing it through a softmax.
if compute_gradient
  [P, P_meta] = SoftmaxForward(H);
else
  P = SoftmaxForward(H);
end

% Fit a bivariate Von Mises distribution to the PDF to get an estimate of the
% center of mass of the PDF, as well as a covariance matrix, and the partial
% derivative of the mean and covariance matrix.
if compute_gradient
  [mu_idx, Sigma_idx, dmu_idx_P, dSigma_idx_P] = FitBivariateVonMises(P);
else
  [mu_idx, Sigma_idx] = FitBivariateVonMises(P);
end

if params.TRAINING.FORCE_ISOTROPIC_VON_MISES
  % Whiten a convariance matrix while preserving its trace.
  Sigma_idx = eye(2) * mean(diag(Sigma_idx));
  % Note that the gradients dSigma_idx_P are now incorrect. This is
  % rectified later during backpropagation.
end

if strcmp(params.HISTOGRAM.VON_MISES_DIAGONAL_MODE, 'clamp');
  assert(params.TRAINING.FORCE_ISOTROPIC_VON_MISES);
  % Clamp the diagonal of the variance to be at least EPS. This is only
  % supported for diagonal von Mises distributions.
  if (Sigma_idx(1) <= params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS)
    Sigma_idx = eye(2) * params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS;
    sigma_clamped = true;
  else
    sigma_clamped = false;
  end
elseif strcmp(params.HISTOGRAM.VON_MISES_DIAGONAL_MODE, 'pad');
  % Add a constant to the diagonal of fitted Von Mises distributions to help
  % generalization. The constant should be at least 1/12 (the variance of a
  % unit-size box) but larger values regularize the Von Mises to be wider, which
  % appears to help.
  Sigma_idx = Sigma_idx + params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS * eye(2);
else
  assert(0)
end

[state_obs.mu, state_obs.Sigma] = IdxToUv(mu_idx, Sigma_idx, params);

if params.DEBUG.GRAY_WORLD_UNWRAPPING
  avg_uv = RgbToUv(avg_rgb);
  deltas = state_obs.mu - avg_uv;
  width = params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE;
  state_obs.mu = state_obs.mu - width * round(deltas / width);
end

if compute_gradient
  dmu_P = cellfun(@(x) x * params.HISTOGRAM.BIN_SIZE, dmu_idx_P, ...
    'UniformOutput', false);
  dSigma_P = cellfun(@(x) x * params.HISTOGRAM.BIN_SIZE.^2, dSigma_idx_P, ...
    'UniformOutput', false);
end

if strcmp(params.HISTOGRAM.VON_MISES_DIAGONAL_MODE, 'clamp');
  % If the gradient has been clamped, then the derivative with respect to Sigma
  % should be set to zero.
  if (sigma_clamped)
    dSigma_P{1} = 0;
    dSigma_P{2} = 0;
    dSigma_P{3} = 0;
  end
end

meta.P = P;

% The entropy of the Von Mises PDF.
meta.entropy = 0.5 * log(det(state_obs.Sigma));

% Because Sigma has a scaled identity matrix added to it, we can derive a lower
% bound on entropy.
meta.minimum_entropy = ...
      0.5 * log(det(params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS ...
      * (params.HISTOGRAM.BIN_SIZE.^2) * eye(2)));

% This is a reasonable "confidence" measurement, in that it is a value in (0, 1]
% where a large value means that the model predicted a very tightly localized
% white point. Equivalent to:
%   confidence = exp(-entropy) / exp(-minimum_entropy).
meta.entropy_confidence = exp(meta.minimum_entropy - meta.entropy);

sub_loss = struct();
sub_losses = struct();
if isfield(params, 'loss_mult')
  sub_loss.loss = 0;
  d_loss_H = 0;
end

if ~isempty(Y)
  % Here we compute two losses: a convex loss (cross-entropy on a discrete label
  % set, useful for pre-training) and a non-convex loss that produces higher
  % quality results when/if a good minimum is found (Von Mises log-likelihood,
  % useful for fine-tuning).

  % This convex loss is just logistic regression between P and a ground-truth
  % PDF indicating the location of the white point.

  if params.TRAINING.SMOOTH_CROSS_ENTROPY
    % Construct a one-hot vector from the ground truth.
    Y_idx = UvToIdx(Y, params);
    P_true = sparse(Y_idx(1), Y_idx(2), true, size(H,1), size(H,2));
  else
    % Construct a PDF from the ground truth.
    P_true = UvToP(Y, params);
  end

  % Compute cross-entropy (aka log-loss) and its gradient.
  log_P = P_meta.H_shifted - log(P_meta.expH_sum);
  sub_losses.crossent = -full(sum(P_true(:) .* log_P(:)));
  d_crossent_H = P - P_true;

  assert(sub_losses.crossent >= -1e-4);
  sub_losses.crossent = max(0, sub_losses.crossent);

  if isfield(params, 'loss_mult')
    sub_loss.loss = sub_loss.loss ...
      + params.loss_mult.crossent * sub_losses.crossent;
    d_loss_H = d_loss_H + params.loss_mult.crossent * d_crossent_H;
  end

  % Switch between different loss functions on the von mises distribution.
  if strcmp(params.TRAINING.VON_MISES_LOSS, 'expected_squared_error')
    % Computes the expected squared error in UV space.

    % Y_sigma is the the covariance matrix of the error. We use a scale
    % identity matrix (ie, isotropic error) which produces errors that are
    % comparable to the other measures and to RGB angular error.
    Y_sigma = eye(2) / 32;
    inv_Y_sigma = inv(Y_sigma);

    % This math is adopted from Section 0.5 of
    % http://www.cogsci.ucsd.edu/~ajyu/Teaching/Tutorials/gaussid.pdf
    sub_losses.vonmises = sum( ...
      (inv_Y_sigma * (state_obs.mu - Y)) .* (state_obs.mu - Y)) ...
      + trace(inv_Y_sigma * state_obs.Sigma); %#ok<MINV>

    d_vonmises_mu = 2 * (state_obs.mu - Y);
    d_vonmises_Sigma = inv_Y_sigma;

  elseif strcmp(params.TRAINING.VON_MISES_LOSS, 'squared_error')
    % Computes the squared error in UV space.
    sub_losses.vonmises = sum((state_obs.mu - Y).^2);
    d_vonmises_mu = 2 * (state_obs.mu - Y);
    d_vonmises_Sigma = zeros(2, 2);

  elseif strcmp(params.TRAINING.VON_MISES_LOSS, 'likelihood')

    % Compute the log-likelihood of x. Here we treat the bivariate Von Mises
    % distribution in state_obs as a simple multivariate normal, essentially
    % un-wrapping the point from a torus to the least-wrapped Cartesian plane.
    [sub_losses.vonmises, ~, d_vonmises_mu, d_vonmises_Sigma] = ...
      LLMultivariateNormal(Y, state_obs.mu, state_obs.Sigma);

    % Flipping the sign to turn log-likelihoods into losses.
    sub_losses.vonmises = -sub_losses.vonmises;
    d_vonmises_mu = -d_vonmises_mu;
    d_vonmises_Sigma = -d_vonmises_Sigma;

    % Because the Von Mises log-likelihood is computed using covariance matrices
    % which have had a non-zero constant added to the diagonal, we can bound the
    % log-likelihood and use this to shift the loss/log-likelihood such that it
    % is non-negative, which makes optimization easier to reason about.
    % Equivalent to:
    %   vonmises_loss_min = -LLMultivariateNormal( ...
    %     [0; 0], [0; 0], ...
    %     eye(2) * (params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS * ...
    %     params.HISTOGRAM.BIN_SIZE.^2));
    vonmises_loss_min = log(2 * pi * ...
      params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS * params.HISTOGRAM.BIN_SIZE^2);

    sub_losses.vonmises = sub_losses.vonmises - vonmises_loss_min;
  else
    assert(false)
  end

  assert(sub_losses.vonmises >= -1e-4);
  sub_losses.vonmises = max(0, sub_losses.vonmises);

  % Backprop the gradient from the Von Mises onto the PDF.
  if params.TRAINING.FORCE_ISOTROPIC_VON_MISES
    d_var = 0.5 * (d_vonmises_Sigma(1,1) + d_vonmises_Sigma(2,2));
    d_vonmises_P = ...
      bsxfun(@plus, d_var * dSigma_P{1} + d_vonmises_mu(1) * dmu_P{1}, ...
                    d_var * dSigma_P{3} + d_vonmises_mu(2) * dmu_P{2});
  else
  % The partial derivatives in dSigma_P have varied sizes so we add them in
  % the most efficient order.
    d_vonmises_P = bsxfun(@plus, bsxfun(@plus, ...
      (2 * d_vonmises_Sigma(1,2)) * dSigma_P{2}, ...
      (d_vonmises_mu(1) * dmu_P{1} + d_vonmises_Sigma(1,1) * dSigma_P{1})), ...
      (d_vonmises_mu(2) * dmu_P{2} + d_vonmises_Sigma(2,2) * dSigma_P{3}));
  end

  % Backprop the gradient from the PDF to H.
  d_vonmises_H = SoftmaxBackward(d_vonmises_P, P_meta);

  power_loss = ...
    isfield(params.TRAINING, 'VONMISES_POWER') ...
    && (params.TRAINING.VONMISES_POWER ~= 1);

  if power_loss
    assert(params.TRAINING.VONMISES_POWER ~= 0);
    [sub_losses.vonmises, d_vonmises_H] = ...
      ApplyPower(params.TRAINING.VONMISES_POWER, ...
        sub_losses.vonmises, d_vonmises_H);
  end

  if isfield(params, 'loss_mult')
    sub_loss.loss = sub_loss.loss ...
      + params.loss_mult.vonmises * sub_losses.vonmises;
    d_loss_H = d_loss_H + params.loss_mult.vonmises * d_vonmises_H;
  end

  if isfield(params, 'loss_mult')
    sub_loss.d_loss_F_fft =  ...
      bsxfun(@times, conj(X_fft), fft2((1 / size(F_fft,1)^2) * d_loss_H));
    if params.TRAINING.LEARN_BIAS
      sub_loss.d_loss_B = d_loss_H;
    end
  end
end

if nargout >= 5
  if (params.RUNTIME.FRAMES_PER_SECOND > 0) ...
      && ~isinf(params.RUNTIME.KALMAN_NOISE_VARIANCE)
    % The motion of the white point over time is assumed to be zero-mean
    % Gaussian noise with an isotropic covariance matrix, whose diagonal
    % elements are some assumed variance constant divided by the frame rate.
    Sigma_noise = (params.RUNTIME.KALMAN_NOISE_VARIANCE ...
      / params.RUNTIME.FRAMES_PER_SECOND) * eye(2);

    % Update the current state's estimate of mu and Sigma using the observed
    % mu and Sigma and the assumed transition model.
    state_next = KalmanUpdate(state_obs, state_last, Sigma_noise);
  else
    % If we're ignoring temporal smoothness, then just set the state to be
    % equal to the observation.
    state_next = state_obs;
  end
end
