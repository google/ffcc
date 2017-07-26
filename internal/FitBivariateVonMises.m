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

function [mu, Sigma, dmu_P, dSigma_P, P_fit] = FitBivariateVonMises(P)
% Given a 2D PDF histogram (PMF), approximately fits a bivariate Von Mises
% distribution to that PDF by computing the local moments. This produces a
% center of mass of the PDF, where the PDF is assumed to lie on a torus rather
% than a cartesian space.
%
% Outputs:
%   mu - the 2D location of the center of the bivariate Von Mises distribution,
%        which is in 1-indexed coordinates of the input PDF P.
%   Sigma - The covariance matrix which was used to compute "confidence".
%   dmu_P - the partial derivatives of mu as a function of P.
%   dSigma_P - the partial derivatives of Sigma as a function of P, where:
%              dSigma_P{1} = d_Sigma(1,1),
%              dSigma_P{2} = Sigma(1,2) = Sigma(2,1), and
%              dSigma_P{3} = Sigma(2,2).
%   P_fit - A 2D PDF histogram like P, showing the fitted distribution uses
%           an ad hoc Gaussian on a torus.

assert(size(P,1) == size(P,2))

n = size(P,1);
assert((n/2) == round(n/2))

angle_scale = n / (2*pi);
angle_step = 1/angle_scale;
angles = (0 : angle_step : (2*pi-angle_step))';

% Fit the mean of the distribution by finding the first moments of of the
% histogram on both axes, which can be done by finding the first moment of the
% sin and cosine of both axes and computing the arctan. Taken from Section 6.2
% of "Bayesian Methods in Structural Bioinformatics", but adapted to a
% histogram. This process can be optimized by computing the rowwise and
% columnwise sums of P and finding the moments on each axis independently.
P1 = sum(P,2);
P2 = sum(P,1)';

sin_angles = sin(angles);
cos_angles = cos(angles);

y1 = sum(P1 .* sin_angles);
x1 = sum(P1 .* cos_angles);
y2 = sum(P2 .* sin_angles);
x2 = sum(P2 .* cos_angles);
mu1 = mod(atan2(y1, x1), 2*pi) * angle_scale;
mu2 = mod(atan2(y2, x2), 2*pi) * angle_scale;
mu = [mu1; mu2] + 1; % 1-indexing

if nargout >= 3
  dmu1_P1 = ((x1*sin_angles - y1*cos_angles) ./ (x1.^2 + y1.^2)) * angle_scale;
  dmu2_P2 = ((x2*sin_angles - y2*cos_angles) ./ (x2.^2 + y2.^2)) * angle_scale;
  dmu_P = {dmu1_P1, dmu2_P2'};
end

if nargout >= 2
  % Fit the covariance matrix of the distribution by finding the second moments
  % of the angles with respect to the mean. This can be done straightforwardly
  % using the definition of variance, provided that the distance from the mean
  % is the minimum distance on the torus. This can become innacurate if the true
  % distribution is very large with respect to the size of the histogram.
  bins = [1:size(P,1)]';
  wrap = @(x) (mod(x + size(P,1)/2 - 1, size(P,1)) + 1);
  wrapped1 = wrap(bins - round(mu(1)));
  wrapped2 = wrap(bins - round(mu(2)));

  E1 = sum(P1 .* wrapped1);
  E2 = sum(P2 .* wrapped2);
  Sigma1 = sum(P1 .* wrapped1.^2) - E1.^2;
  Sigma2 = sum(P2 .* wrapped2.^2) - E2.^2;
  Sigma12 = sum(sum(P .* bsxfun(@times, wrapped1, wrapped2'))) - E1 * E2;

  Sigma = [Sigma1, Sigma12; ...
           Sigma12, Sigma2];
end

if nargout >= 4
  dSigma1_P1 = wrapped1 .* (wrapped1 - 2*E1);
  dSigma2_P2 = wrapped2 .* (wrapped2 - 2*E2);
  dSigma12_P = bsxfun(@times, wrapped1 - E1, wrapped2' - E2) - E1 * E2;
  dSigma_P = {dSigma1_P1, dSigma12_P, dSigma2_P2'};
end

if nargout >= 5
  % A predicted PDF corresponding to the fitted mu and Sigma, computed using
  % a cheap approximation to the brute force PDF, which is an infinite sum of
  % the Gaussian and all possible wrapped Gaussians around the torus. Here we
  % just wrap the Gaussian once, by shifting mu by one period in every
  % direction (8 in total) around the torus, and assign each bin the minimum
  % mahalanobis distance over all possible shifts, and use that exponentiated
  % distance to produce a normalized PDF. This behaves well and does not have
  % numerical issues when the distribution's concentration is small, though
  % it becomes innacurate when the concentration is large. This approach
  % also makes things easier to differentiate.
  [i_bins,j_bins] = ndgrid(1:size(P,1), 1:size(P,2));
  X = [i_bins(:), j_bins(:)];
  inv_Sigma = inv(Sigma);
  D_min = inf(size(P));
  for oi = -1:1
    for oj = -1:1
      Xc = bsxfun(@minus, X, mu' + size(P,1) * [oi, oj]);
      D = reshape(sum(Xc .* (Xc * inv_Sigma),2), size(P));
      D_min = min(D_min, D);
    end
  end
  P_fit = exp(-D_min/2);
  P_fit = P_fit / sum(P_fit(:));
end
