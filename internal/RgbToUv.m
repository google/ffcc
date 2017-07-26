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

function uv = RgbToUv(rgb)
% Turns an RGB matrix (3xn) specifying the color of illuminants
% into a UV matrix (2xn) specifying the chroma of the white point of
% the scene (where the white point is 1/illuminant).

assert(size(rgb,1) == 3);
uv = [log(rgb(2,:) ./ rgb(1,:)); log(rgb(2,:) ./ rgb(3,:))];
