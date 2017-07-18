function rgb = UvToRgb(uv)
% Turns a UV matrix (2xn) specifying the chromas of white points into a RGB
% matrix (3xn) where each column is unit-norm, specifying the colors of the
% illuminants (where illuminant is 1/white-point).

assert(size(uv,1) == 2);

rgb = [exp(-uv(1,:)); ...
       ones(1, size(uv,2));
       exp(-uv(2,:))];
rgb = bsxfun(@rdivide, rgb, sqrt(sum(rgb.^2,1)));
