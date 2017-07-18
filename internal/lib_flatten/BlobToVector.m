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

function [vector, metadata] = BlobToVector(blob)
% Serializes any n-dimensional matrix, struct, or cell array into a single
% vector (assuming the elements of the structs or cell arrays are
% themselves n-dimensional matrices, structs, or cell arrays). Also returns
% the metadata necessary to reconstruct the blob from the vector using
% VectorToBlob().

metadata = struct();
if isnumeric(blob) || islogical(blob)
  metadata.class = 'numeric';
  metadata.numeric_class = class(blob);
  metadata.size = size(blob);
  vector = double(blob(:));
elseif iscell(blob)
  metadata.class = 'cell';
  metadata.metadatas = cell(size(blob));
  metadata.counts = zeros(size(blob));
  vector = {};
  for i = 1:numel(blob)
    sub_blob = blob{i};
    [sub_vector, sub_metadata] = BlobToVector(sub_blob);
    vector{end+1} = double(sub_vector(:));
    metadata.metadatas{i} = sub_metadata;
    metadata.counts(i) = length(sub_vector);
  end
  vector = cat(1,vector{:});
elseif isstruct(blob)
  metadata.class = 'struct';
  metadata.fieldnames = fieldnames(blob);
  metadata.metadatas = struct();
  metadata.counts = struct();
  fields = fieldnames(blob)';
  vector = {};
  for i_field = 1:length(fields)
    field = fields{i_field};
    sub_blob = blob.(field);
    [sub_vector, sub_metadata] = BlobToVector(sub_blob);
    vector{i_field} = sub_vector;
    metadata.metadatas.(field) = sub_metadata;
    metadata.counts.(field) = length(sub_vector);
  end
  vector = cat(1,vector{:});
else
  assert(0);
end
