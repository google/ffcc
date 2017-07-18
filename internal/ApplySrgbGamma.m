function im = ApplySrgbGamma(im_linear)
% Applies the SRGB gamma curve. Adapted from
% https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_.28CIE_xyY_or_CIE_XYZ_to_sRGB.29

assert(isa(im_linear, 'double'))
assert(max(im_linear(:)) <= 1)
assert(min(im_linear(:)) >= 0)

a = 0.055;
t = 0.0031308;
im = (im_linear * 12.92) .* (im_linear <= t) ...
   + ((1 + a)*im_linear.^(1/2.4) - a) .* (im_linear > t);
