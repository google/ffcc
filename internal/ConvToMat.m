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

function A = ConvToMat(sz, F, edge_behavior)
% Given some 2D image size sz, some 2D filter F, and some edge behavior,
% returns a matrix A such that A*vec(x) = vec(conv(x, f)).

% Flip the filter, such that conv(x, F) = cross-correlation(x, F_mirror)
F_mirror = reshape(F(end:-1:1), size(F));

% The rows of the A matrix.
[i0, j0] = ndgrid(1:sz(1), 1:sz(2));
i0 = i0(:);
j0 = j0(:);
base_idx0 = sub2ind(sz, i0, j0);

% The possible offsets in i and j that F requires.
i_radius = (size(F,1)-1)/2;
i_offsets = -floor(i_radius):ceil(i_radius);
j_radius = (size(F,2)-1)/2;
j_offsets = -floor(j_radius):ceil(j_radius);

A_idx = {};
for i_offset = i_offsets
  for j_offset = j_offsets

    % The columns of the A matrix.
    i1 = i0 + i_offset;
    j1 = j0 + j_offset;

    % Depending on the edge behavior, the i1/j1 coordinates need to be
    % modified or selectively omitted.
    if strcmp(edge_behavior, 'zero')
      keep = (i1 >= 1) & (j1 >= 1) & (i1 <= sz(1)) & (j1 <= sz(2));
      idx0 = base_idx0(keep);
      idx1 = sub2ind(sz, i1(keep), j1(keep));
    elseif strcmp(edge_behavior, 'replicate')
      i1 = min(max(i1, 1), sz(1));
      j1 = min(max(j1, 1), sz(2));
      idx0 = base_idx0;
      idx1 = sub2ind(sz, i1, j1);
    elseif strcmp(edge_behavior, 'circular')
      i1 = mod(i1-1, sz(1))+1;
      j1 = mod(j1-1, sz(2))+1;
      idx0 = base_idx0;
      idx1 = sub2ind(sz, i1, j1);
    else
      assert(false); % Unsupported edge behavior?
    end

    % Accumulate (i,j,v) triplets with which we will construct a sparse matrix.
    f = F_mirror(i_offset+floor(i_radius)+1, j_offset+floor(j_radius)+1);
    A_idx{end+1} = [idx0, idx1, repmat(f, length(idx0), 1)];
  end
end
A_idx = cat(1,A_idx{:});

% Construct the matrix.
A = sparse(A_idx(:,1), A_idx(:,2), A_idx(:,3), prod(sz), prod(sz));
