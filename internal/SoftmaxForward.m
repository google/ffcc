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

function [P, meta] = SoftmaxForward(H, dims)
% Applies a softmax function to H with respect to the dimensions enumerated in
% dims to produce a normalized tensor P. If dims is not specified then the
% softmax is computed over all dimensions. The necessary bookkeeping for
% backpropagation is stored in the optional output argument "meta".
% Example usage:
%   SoftmaxForward(X) - The output is single PDF which sums to 1 across all
%     dimensions of X.
%   SoftmaxForward(X, [2,3]) - Assuming X is a 3d tensor, the output is many
%     PDFs where each row sums to 1.

% dims is optional, but must be a row vector if specified.
if nargin < 2
  dims = 1:ndims(H);
else
  assert(size(dims,2) == numel(dims))
end

% Take the max over H in all dimensions specified.
max_val = H;
for dim = dims
  max_val = max(max_val, [], dim);
end

% Shift and exponentiate H.
H_shifted = bsxfun(@minus, H, max_val);
expH = exp(H_shifted);

% Take the sum over the un-normalized histogram in all dimensions specified.
expH_sum = expH;
for dim = dims
  expH_sum = sum(expH_sum, dim);
end

% Normalize.
P = bsxfun(@rdivide, expH, expH_sum);

% Bookkeeping for the backward pass.
meta.dims = dims;
meta.P = P;
meta.expH_sum = expH_sum;
meta.expH = expH;
meta.H = H;
meta.H_shifted = H_shifted;
