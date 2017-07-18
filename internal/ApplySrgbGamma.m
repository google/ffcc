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
