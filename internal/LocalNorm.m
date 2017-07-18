function im_norm = LocalNorm(im, weight, radius, rho)
% Computes a local norm of an input image with respect to some weight/weight,
% window half-width, and power.
% This code only exists to support ChannelizeImage2015, which is legacy code
% included for comparison and paper-writing.

assert(all(im(:) >= 0))

if isempty(weight)
  weight = true(size(im,1), size(im,2));
end
weight = double(weight);

% Precompute the factor that all box filters must be divided by.
bias = 1 ./ max(eps, BoxFilter(weight, radius));

im_norm = {};
for c = 1:size(im,3)
  % For each pixel, compute E[x^rho]^(1/rho) using the box filter to efficiently
  % evaluate the expectation for each pixel simultaneously.
  im_norm{c} = weight .* max(0, ...
    BoxFilter(weight .* max(0, im(:,:,c)).^rho, radius) .* bias).^(1/rho);
end
im_norm = cat(3, im_norm{:});

assert(~any(isnan(im_norm(:))))
assert(~any(isinf(im_norm(:))))
