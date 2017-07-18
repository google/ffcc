function im_std = LocalStandardDeviation(im, weight, radius)
% Computes a local standard deviation of an input image with respect to some
% weight/mask and window half-width.
% This code only exists to support ChannelizeImage2015, which is legacy code
% included for comparison and paper-writing.

assert(all(im(:) >= 0))

if isempty(weight)
  weight = true(size(im,1), size(im,2));
end
weight = double(weight);

% Precompute the factor that all box filters must be divided by.
bias = 1 ./ max(eps, BoxFilter(weight, radius));

im_std = {};
for c = 1:size(im,3)
  % Compute the local expectation of the image.
  EX = BoxFilter(weight .* im(:,:,c), radius) .* bias;
  % Compute the local expectation of the image^2
  EX_rho = BoxFilter(weight .* max(eps, im(:,:,c)).^2, radius) .* bias;
  % Compute the square root of the local variance using the equivalence:
  %   Var[x] = E[x^2] - E[x]^2
  im_std{c} = sqrt(max(0, abs(EX_rho - max(0, EX).^2)));
end
im_std = cat(3, im_std{:});

assert(~any(isnan(im_std(:))));
assert(~any(isinf(im_std(:))));
