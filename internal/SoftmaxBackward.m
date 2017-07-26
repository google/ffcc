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

function dH = SoftmaxBackward(dP, meta)
% Takes a gradient with respect to P (the output of SoftmaxForward) and
% backpropagates it onto H (the input to SoftmaxForward) using the metadata
% returned by SoftmaxForward.

d_expH = bsxfun(@rdivide, dP, meta.expH_sum);
d_sum = d_expH .* meta.P;
for dim = meta.dims
  d_sum = sum(d_sum, dim);
end
d_sum = full(d_sum);
dH = bsxfun(@minus, d_expH, d_sum) .* meta.expH;
