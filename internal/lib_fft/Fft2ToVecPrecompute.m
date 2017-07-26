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

function mapping = Fft2ToVecPrecompute(sz)
% Given a (square) image size sz as input, precomputes a mapping that can be
% used to quickly convert filter/images of size sz to vectors, and back.
% This mapping can be used as an optional input to Fft2ToVec() and VecToFft2().

% The input must be the size of a single square image.
assert(numel(sz) == 2)
assert(sz(1) == sz(2));

n = prod(sz);

assert(n > 1)
assert(log2(n) == round(log2(n)))

% We can find the mapping by making a real and imaginary image where each value
% is unique and known a-prior (here's it's just integers from 1 to n) and then
% seeing where they end up in the vector representation.
V_real = Fft2ToVec(reshape(1:n, sz), [], false);
V_imag = Fft2ToVec(1i * reshape(1:n, sz), [], false);

mapping.to_vec.real = uint32(V_real(V_real > 0));
mapping.to_vec.imag = uint32(V_imag(V_imag > 0));

% Here we run the same process in reverse.
F_idx = VecToFft2([1:n]', [], false);
mapping.to_fft.real = uint32(real(F_idx));
mapping.to_fft.imag = uint32(max(1, abs(imag(F_idx))));
mapping.to_fft.imag_sign = sign(imag(F_idx));
