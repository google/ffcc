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

function im_channels = ChannelizeImage(im, mask)
% Generate feature images (color and gradient) and combine them into
% different cell channels.

assert(isa(mask, 'logical'));

im_channels = {};

im_channels{1} = cast(bsxfun(@times, double(im), mask), 'like', im);
im_channels{2} = cast(MaskedLocalAbsoluteDeviation(im, mask), 'like', im);
