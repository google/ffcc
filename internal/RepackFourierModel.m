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

function [ output ] = RepackFourierModel( F_fft )
% Repack the Fourier coefficients to be compatible with the Halide
% implementation format

half_fft_size = size(F_fft, 1) / 2 + 1;

output = {};
for c = 1:size(F_fft, 3)
  output{end + 1} = single(real(F_fft(1:half_fft_size, :, c)));
  output{end + 1} = single(imag(F_fft(1:half_fft_size, :, c)));
end

  output = cat(3, output{:});
end

