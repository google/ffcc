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

function LLMultivariateNormal_Test

rng('default')

kNumTrials = 128;

for i_rand = 1:kNumTrials

  % Pick a random dimensionality between 2 and 10.
  ndims = ceil(1+rand*10);

  % Randomly generate a mean and diagonal covariance matrix.
  mu = randn(ndims, 1);
  r = randn(ndims,ndims);
  r = (r + r')/2;
  Sigma = expm(r) + eye(ndims);

  % Randomly generate several points.
  X = randn(ndims, 10);

  % Compute a ground truth likelihood under the multivariate normal
  % PDF code provided by Matlab.
  P = mvnpdf(X', mu', Sigma);

  % Compute our log-likelihood.
  LL = LLMultivariateNormal(X, mu, Sigma);

  % Assert that our (non-log) likelihoods is very close to Matlab's.
  assert( max(abs(exp(LL) - P')) < 1e-12 )

  % Randomly generate one point, and compute it's log-likelihood and
  % partial derivatives.
  X = randn(ndims, 1);
  [LL, dLL_X, dLL_mu, dLL_Sigma] = LLMultivariateNormal(X, mu, Sigma);

  % Numerically approximate the partial derivative as a function of X.
  nLL_X = nan(size(dLL_X));
  ep = 1e-5;
  for d = 1:numel(X)
    X_ = X;
    X_(d) = X_(d) + ep;
    LL_ = LLMultivariateNormal(X_, mu, Sigma);
    nLL_X(d) = (LL_ - LL) / ep;
  end

  % Check that the analytical derivative is close.
  assert(max(abs(dLL_X - nLL_X)) < 1e-4)

  % Numerically approximate the partial derivative as a function of mu.
  nLL_mu = nan(size(dLL_mu));
  ep = 1e-5;
  for d = 1:numel(mu)
    mu_ = mu;
    mu_(d) = mu_(d) + ep;
    LL_ = LLMultivariateNormal(X, mu_, Sigma);
    nLL_mu(d) = (LL_ - LL) / ep;
  end

  % Check that the analytical derivative is close.
  assert(max(abs(dLL_mu - nLL_mu)) < 1e-4)

  % Numerically approximate the partial derivative as a function of Sigma.
  nLL_Sigma = nan(size(dLL_Sigma));
  ep = 1e-5;
  for d = 1:numel(Sigma)
    Sigma_ = Sigma;
    Sigma_(d) = Sigma_(d) + ep;
    LL_ = LLMultivariateNormal(X, mu, Sigma_);
    nLL_Sigma(d) = (LL_ - LL) / ep;
  end

  % Check that the analytical derivative is close.
  assert(max(abs(dLL_Sigma(:) - nLL_Sigma(:))) < 1e-4)

end

fprintf('Test Passed\n');
