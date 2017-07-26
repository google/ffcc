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

function [state_next] = KalmanUpdate(state_obs, state_last, Sigma_noise)
% Implements a simple variant of a Kalman filter with no dynamics or
% control, but where each observation has a covariance matrix associated with
% it and where we have a zero-mean multivariate Gaussian model of transition
% noise. Given a Gaussian (mu and Sigma) state_last, which models the current
% position, updates that position with a new observed Gaussian state_obs, while
% taking into account normally distributed noise in the state from one frame to
% the next with covariance matrix Sigma_noise.

% If the last state is empty, then we return the observed state.
if isempty(state_last)
  state_next = state_obs;
  return;
end

% Assert that each "mu" is a dx1 vector, and each Sigma is a dxd matrix.
assert(all(size(state_obs.mu) == size(state_last.mu)));
assert(all(size(state_obs.Sigma) == size(state_last.Sigma)));
assert(all(size(state_obs.Sigma) == size(Sigma_noise)));
assert(size(state_obs.mu,2) == 1);
assert(size(state_obs.mu,1) == size(state_obs.Sigma,1));
assert(size(state_obs.Sigma,1) == size(state_obs.Sigma,2));

% Assert all Sigma matrices are symmetric and PSD.
assert(abs(max(max(state_obs.Sigma - state_obs.Sigma'))) < 1e-8);
assert(abs(max(max(state_last.Sigma - state_last.Sigma'))) < 1e-8);
assert(abs(max(max(Sigma_noise - Sigma_noise'))) < 1e-8);
assert(det(state_obs.Sigma) >= 0)
assert(det(state_last.Sigma) >= 0)
assert(det(Sigma_noise) >= 0)

% The current estimate of state's covariance is updated with the covariance of
% the noise model. This is equivalent to convolving the last Gaussian with
% the (zero-mean) Gaussian representing the transition "noise".
state_last.Sigma = state_last.Sigma + Sigma_noise;

% the last state "state_last" is updated by simply summing the log-likelihoods
% described by the state and observation in "standard form", and then
% converting them back into mu+Sigma form. For speed and simplicity we
% special-case the update when the dimensionality is 1 or 2.
if numel(state_obs.mu) == 1

  state_next.Sigma = 1 ./ (1 ./ state_obs.Sigma + 1 ./ state_last.Sigma);
  state_next.mu = state_next.Sigma ...
                * (state_obs.mu ./ state_obs.Sigma ...
                   + state_last.mu ./ state_last.Sigma);

elseif numel(state_obs.mu) == 2

  % This is the closed form for performing a 2x2 matrix inverse.
  inv2 = @(x)( [x(2,2), -x(1,2); -x(2,1), x(1,1)] ...
             / (x(1,1) * x(2,2) - x(1,2) * x(2,1)));
  iSigma_obs = inv2(state_obs.Sigma);
  iSigma_last = inv2(state_last.Sigma);
  state_next.Sigma = inv2(iSigma_obs + iSigma_last);
  state_next.mu = state_next.Sigma ...
                * (iSigma_obs * state_obs.mu + iSigma_last * state_last.mu);

else

  iSigma_obs = inv(state_obs.Sigma);
  iSigma_last = inv(state_last.Sigma);
  state_next.Sigma = inv(iSigma_obs + iSigma_last);
  state_next.mu = state_next.Sigma ...
                * (iSigma_obs * state_obs.mu + iSigma_last * state_last.mu);

end

