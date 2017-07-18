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

function ConvToMat_Test
% Tests that ConvToMat returns identical results to conv() and imfilter().

for i_rand = 1:64
  X_sz = [ceil(rand * 64), ceil(rand * 64)];
  X = randn(X_sz);
  f = randn(ceil(rand * 6), ceil(rand * 6));
  A = ConvToMat(X_sz, f, 'zero');
  err = conv2(X, f, 'same') - reshape(A * X(:), size(X));
  assert(max(abs(err(:))) < 1e-10)

  A = ConvToMat(X_sz, f, 'replicate');
  err = imfilter(X, f, 'replicate', 'same', 'conv') - reshape(A * X(:), size(X));
  assert(max(abs(err(:))) < 1e-10)

  A = ConvToMat(X_sz, f, 'circular');
  err = imfilter(X, f, 'circular', 'same', 'conv') - reshape(A * X(:), size(X));
  assert(max(abs(err(:))) < 1e-10)
end

fprintf('Test Passed\n');
