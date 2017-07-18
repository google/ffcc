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

function rgb = UvToRgb(uv)
% Turns a UV matrix (2xn) specifying the chromas of white points into a RGB
% matrix (3xn) where each column is unit-norm, specifying the colors of the
% illuminants (where illuminant is 1/white-point).

assert(size(uv,1) == 2);

rgb = [exp(-uv(1,:)); ...
       ones(1, size(uv,2));
       exp(-uv(2,:))];
rgb = bsxfun(@rdivide, rgb, sqrt(sum(rgb.^2,1)));
