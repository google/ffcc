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
