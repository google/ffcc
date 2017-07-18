function im_linear = UndoSrgbGamma(im)
% Undoes the SRGB gamma curve, re-linearizing SRGB data. Adapted from
% https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation

assert(isa(im, 'double'))
assert(max(im(:)) <= 1)
assert(min(im(:)) >= 0)

a = 0.055;
t = 0.04045;
im_linear = (im ./ 12.92) .* (im <= t) ...
          + ((im + a) ./ (1 + a)).^2.4 .* (im > t);
