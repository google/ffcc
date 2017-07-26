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

function [LL, dLL_X, dLL_mu, dLL_Sigma] = LLMultivariateNormal(X, mu, Sigma)
% Given a dxn matrix of points X, a dx1 vector mu, and a dxd covariance matrix
% Sigma, returns the log-likelihood of each column of X under the multivariate
% Gaussian defined by mu and Sigma. Optionally returns the partial
% derivatives of the log-likelihood as a function of X, mu, and Sigma.

assert(size(mu,2) == 1)
assert(size(X,1) == size(mu,1))
assert(size(mu,1) == size(Sigma,1))
assert(size(Sigma,1) == size(Sigma,2))

ndims = size(X,1);
ndata = size(X,2);

% Center X around the mean mu.
Xc = bsxfun(@minus, X, mu);
% Multiply the centered X by the inverse of the covariance matrix
iSigma = inv(Sigma);
Xc_iSigma = iSigma * Xc;

% Compute the log of the partition function.
logZ = 0.5*(ndims * log(2*pi) + log(det(Sigma)));

% Compute the log-likelihood.
LL = -logZ - 0.5 * sum(Xc_iSigma .* Xc,1);

% Optionally compute the partial derivatives.
if nargout >= 2
  dLL_X = -Xc_iSigma;
end

if nargout >= 3
  dLL_mu = Xc_iSigma;
end

if nargout >= 4
  dLL_Sigma = bsxfun(@minus, bsxfun(@times, ...
    reshape(Xc_iSigma, ndims, 1, ndata), ...
    reshape(Xc_iSigma, 1, ndims, ndata)), iSigma) / 2;
  % the dLL_Sigma definition is equivalent to:
  % dLL_Sigma(:,:,i) = (Xc_iSigma(:,i) * Xc_iSigma(:,i)' - iSigma)/2;
  % for each i, but is vectorized and therefore faster.
end
