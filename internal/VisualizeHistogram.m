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

function output = VisualizeHistogram(X, params, rho)

if ~exist('rho', 'var')
  rho = 0.5;
end

if any(X(:) < 0)
  max_val = max(max(abs(X), [], 1), [], 2);
  X = bsxfun(@rdivide, X, max_val);
  X = sign(X) .* (abs(X).^rho);
  X = (X + 1) / 2;
else
  X = bsxfun(@rdivide, X, max(max(X, [], 1), [], 2));
  X = X.^rho;
end

bins = EnumerateBins(params);

assert(length(bins) == size(X,1))
assert(length(bins) == size(X,2))

[u, v] = ndgrid(bins, bins);
log_rgb = cat(3, -u, zeros(size(u)), -v);
rgb = exp(bsxfun(@minus, log_rgb, max(log_rgb, [], 3)));
rgb = bsxfun(@rdivide, rgb, max(rgb,[],3));

zero_bin_idx = find(bins == 0);
if (numel(zero_bin_idx) >= 1)
  assert(numel(zero_bin_idx) == 1);
  X(zero_bin_idx,:,:) = 1;
  X(:,zero_bin_idx,:) = 1;
end

V = {};
for c = 1:size(X,3)
  V{c} = bsxfun(@times, X(:,:,c), rgb);
  if c < size(X,3)
    V{c} = padarray(V{c}, [2,0], 1, 'post');
  end
end
V = cat(1, V{:});

if nargout == 0
  imagesc(V);
else
  output = V;
end
