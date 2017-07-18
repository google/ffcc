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

function transformation = ....
  ComputeWhiteningTransformation(X, subtract_mean, variance_pad, mode)
% Computes a whitening transformation from input matrix X. Instead of computing
% whitened data, this function instead computes an affine transformation which
% causes the input data to be whitened.
% Inputs:
%   X is d x n (ie, each column is a datapoint).
%   subtract_mean=true causes the data to be centered before whitening.
%   variance_pad is a float which is added to the diagonal of the covariance.
%   mode can be either 'zca' or 'pca', depending on the desired whitening.
% The output is a struct which contains function handles to whiten and unwhiten
% any input matrix, as well as the matrices that define thoe transformations.

X = double(X);

if subtract_mean
  mu = mean(X, 2);
  X_centered = bsxfun(@minus, X, mu);
else
  mu = zeros(size(X, 1), 1);
  X_centered = X;
end

Sigma = (X_centered * X_centered') / size(X_centered,2);
Sigma = Sigma + variance_pad * eye(size(Sigma));

% Use SVD to more-stably compute eigenvectors from the symmetric Sigma.
[U, S, ~] = svd(Sigma);  % No need to compute V.
S = max(eps, diag(S));

assert(all(S >= (variance_pad - 1e-8)))

inv_sqrt_S = sqrt(1./S);
inv_sqrt_S_mat = sparse(1:length(inv_sqrt_S), 1:length(inv_sqrt_S), inv_sqrt_S);

if strcmp(mode, 'zca')
  % "ZCA-style" whitening.
  A = U * inv_sqrt_S_mat * U';
elseif strcmp(mode, 'pca')
  % "PCA-style" whitening.
  A = inv_sqrt_S_mat * U';
else
  assert(false);
end
A = full(A);

b = A * -mu;
inv_A = inv(A);

transformation.mu = mu;
transformation.Sigma = Sigma;
transformation.A = A;
transformation.inv_A = inv_A;
transformation.b = b;
transformation.whiten = @(x)(bsxfun(@plus, A * x, b));
transformation.unwhiten = @(x)(inv_A * bsxfun(@minus, x, b));
