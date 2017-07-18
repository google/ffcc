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

function [rgbC, alphaC] = AlphaMatte(rgbA, alphaA, rgbB, alphaB)
% Apply an "over" operator, where (rgbA, alphaA) is overlayed on (rgbB, alphaB).
% The output is a composite image and alpha matte (rgbC, alphaC).

rgbC = bsxfun(@rdivide, ...
  bsxfun(@times, rgbA, alphaA) + bsxfun(@times, rgbB, alphaB .* (1 - alphaA)), ...
  alphaA + alphaB .* (1 - alphaA));
alphaC = alphaA + alphaB .* (1 - alphaA);
