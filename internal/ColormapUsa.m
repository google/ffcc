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

function output = ColormapUsa
% Produces a colormap that goes blue-white-red. Useful for visualizing
% images where negative, positive, and zero values have qualitatively
% different meanings.
%
% If no output arguments are specified, the current figure's colormap is
% set. If an output argument is specified, then the colormap is returned
% and the figure is not modified.
%
% Usage example:
%   imagesc(image, [-1, 1]); ColormapUsa;

n = 512;
x = [0:n]'/n;

b = 1 + min(0, 1 - 2*x);
r = 1 + min(0, 1 - 2*(1-x));
cmap = [r, min(r, b), b];

if nargout == 0
  colormap(cmap);
else
  output = cmap;
end
