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

function FitBivariateVonMises_Test
rng('default')

kNumTrials = 16;
kHistogramSize = 64;

errs_mu = nan(kNumTrials,1);
errs_sigma = nan(kNumTrials,1);
for i_rand = 1:kNumTrials

  % Randomly generate a mean and diagonal covariance matrix.
  mu_true = 1 + ceil(rand(2, 1)*(kHistogramSize-2));
  r = randn(2,2);
  r = (r + r')/2;
  Sigma_true = expm(r) + 3*eye(2);

  % Compute a histogram with that mean and covariance, by construction a delta
  % function at the mean and convolving it with a centered filter with the
  % true covariance.
  P = zeros(kHistogramSize, kHistogramSize);
  P(mu_true(1), mu_true(2)) = 1;
  [i,j] = ndgrid(-(2*kHistogramSize):(2*kHistogramSize), ...
                 -(2*kHistogramSize):(2*kHistogramSize));
  f_true = reshape(mvnpdf([i(:), j(:)], [0, 0], Sigma_true), size(i));
  P = imfilter(P, f_true, 'same', 'circular');
  P = P ./ sum(P(:));

  % Estimate the mean and covariance.
  [mu, Sigma, ~, ~, P_fit] = FitBivariateVonMises(P);

  try
    figure(1); imagesc([P, P_fit]);
    axis image off;
    drawnow;
  catch me
    me;
  end

  errs_mu(i_rand) = sum(abs((mu - mu_true)));

  % This is a reasonable measure of the difference between two 2x2 PSD
  % matrices.
  errs_sigma(i_rand) = abs(log(det(inv(Sigma) * Sigma_true)));

end

fprintf('Mu Errors:    ');
fprintf('%0.6f ', prctile(errs_mu, [20, 50, 90, 95, 99, 100]));
fprintf('\n');

fprintf('Sigma Errors: ');
fprintf('%0.6f ', prctile(errs_sigma, [20, 50, 90, 95, 99, 100]));
fprintf('\n');

assert(max(errs_mu) < 1e-5)
assert(max(errs_sigma) < 1e-3)

% Check that the analytical gradient returned by the function matches what
% we compute with finite differences.
kNumTrials = 4;
kHistogramSize = 8;

for i_rand = 1:kNumTrials
  P = max(0, rand(kHistogramSize, kHistogramSize) - 0.5);
  P = P ./ sum(P(:));

  [mu, Sigma, dmu_P, dSigma_P] = FitBivariateVonMises(P);
  dmu_P = cat(3, ...
    repmat(dmu_P{1}, 1, kHistogramSize), ...
    repmat(dmu_P{2}, kHistogramSize, 1));

  dSigma_P = cat(3, ...
    repmat(dSigma_P{1}, 1, kHistogramSize), ...
    dSigma_P{2}, ...
    repmat(dSigma_P{3}, kHistogramSize, 1));

  nmu_P = nan(size(dmu_P));
  nSigma_P = nan(size(dSigma_P));
  step_size = 1e-5;
  for d = 1:numel(P)
    P_ = P;
    P_ = P_ / sum(P_(:));
    P_(d) = P_(d) + step_size;
    [mu_, Sigma_] = FitBivariateVonMises(P_);
    [i,j] = ind2sub(size(P), d);
    nmu_P(i, j, :) = permute((mu_ - mu) / step_size, [2,3,1]);
    nSigma_P(i, j, :) = permute( ...
      (Sigma_(triu(true(2))) - Sigma(triu(true(2)))) / step_size, [2,3,1]);
  end

  dmu_P_errs = squeeze(max(max(abs(dmu_P - nmu_P))));
  assert(all(dmu_P_errs < 1e-2))

  dSigma_P_errs = max(max(abs(nSigma_P - dSigma_P)));
  assert(all(dSigma_P_errs < 1e-2))
end

fprintf('Test Passed\n');
