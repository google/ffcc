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

function im_edge = LocalAbsoluteDeviation(im)
% Compute Local Absolute Deviation in sliding window fashion. The window
% size is 3x3. If the input image is between [a, b] then the output image
% will be between [0, b-a].

% Upgrade to 16-bit because we have minus here
im_pad = Pad1(int16(im));

im_edge = {};
for c = 1:size(im,3)
  im_edge{c} = 0;
  for oi = -1:1
    for oj = -1:1
      if (oi == 0) && (oj == 0)
        continue
      end
      im_edge{c} = im_edge{c} + ...
                   abs(im_pad([1:size(im,1)] + oi + 1, ...
                              [1:size(im,2)] + oj + 1, c) - ...
                              int16(im(:,:,c)));
    end
  end
end
im_edge = cat(3, im_edge{:});
% Convert back to 8-bit
im_edge = uint8(bitshift(im_edge, -3));
