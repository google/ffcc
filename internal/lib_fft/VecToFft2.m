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

function F = VecToFft2(V, mapping, apply_scaling)
% Takes a K*N*M real-valued tensor V and reconstructs a sqrt(K)*sqrt(K)*N*M
% complex-valued tensor of images/filters in the Fourier domain F, where each
% K-length vector in V corresponds to a square image/filter in F.
% Allows for optional input arguments which specify a faster lookup table,
% precomputed by Fft2ToVecPrecompute().

if ~exist('mapping', 'var')
  mapping = [];
end

if ~exist('apply_scaling', 'var')
  apply_scaling = true;
end

% Currently only two outer dimensions of V are supported.
assert(size(V,1) * size(V,2) * size(V,3) == numel(V));

n = sqrt(size(V,1));
assert(n == round(n))

if size(V,1) ~= numel(V)
  % If V is not a single vector, loop over the outer dimensions of V.
  F = zeros(n, n, size(V,2), size(V,3));
  for d = 1:size(V,3)
    for c = 1:size(V,2)
      F(:, :, c, d) = VecToFft2(V(:, c, d), mapping, apply_scaling);
    end
  end
else
  if ~isempty(mapping) % If a lookup table is given as input, use it.
    if apply_scaling
      V = V / sqrt(2);
    end

    F = complex(V(mapping.to_fft.real), ...
                mapping.to_fft.imag_sign .* V(mapping.to_fft.imag));

    if apply_scaling
      F([1, n/2+1], [1, n/2+1]) = F([1, n/2+1], [1, n/2+1]) * sqrt(2);
    end

  else % Otherwise unpack the FFT manually.
    F_real = zeros(n, n); % Can be initialized to anything.
    F_imag = zeros(n, n); % Must be initialized to 0, some 0's persist.

    if apply_scaling
      V = V / sqrt(2);
    end

    % Fill in the real values from the first half of V.
    F_real(1:(n/2+1), 1) = V(1:(n/2+1));
    F_real(1:(n/2+1), n/2+1) = V(n/2 + 1 + (1:(n/2+1)));
    F_real(:, 2:(n/2)) = reshape(V(n + 2 + (1:(n*(n/2-1)))), n, []);
    s = n*n/2 + 2;

    if apply_scaling
      F_real([1, n/2+1], [1, n/2+1]) = F_real([1, n/2+1], [1, n/2+1]) * sqrt(2);
    end

    % Fill in the imaginary values from the second half of V.
    F_imag(2:(n/2), 1) = V(s + (1:(n/2-1)));
    F_imag(2:(n/2), n/2+1) = V(s + n/2 - 1 + (1:(n/2 - 1)));
    F_imag(:, 2:(n/2)) = reshape(V(s + n - 2 + (1:(n*(n/2-1)))), n, []);

    F = complex(F_real, F_imag);

    F(1,(n/2+2:end)) = fliplr(conj(F(1,(2:(n/2)))));
    F(2:end,(n/2+2:end)) = rot90(conj(F(2:end, 2:(n/2))),2);
    F((n/2+2):end,[1, n/2+1]) = flipud(conj(F(2:(n/2),[1, n/2+1])));
  end
end
