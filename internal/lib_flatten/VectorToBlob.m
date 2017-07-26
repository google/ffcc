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

function blob = VectorToBlob(vector, metadata)
% Takes the vector of serialized values and the struct metadata produced by
% BlobToVector and reconstructs an array, cell array, or struct.

if strcmp(metadata.class, 'numeric')
  blob = cast(reshape(vector, metadata.size), metadata.numeric_class);
elseif strcmp(metadata.class, 'cell')
  count = 0;
  blob = cell(size(metadata.metadatas));
  for i = 1:numel(metadata.metadatas)
    sub_count = metadata.counts(i);
    sub_vector = vector(count + (1:sub_count));
    sub_metadata = metadata.metadatas{i};
    blob{i} = VectorToBlob(sub_vector, sub_metadata);
    count = count + sub_count;
  end
  assert(count == length(vector));
elseif strcmp(metadata.class, 'struct')
  count = 0;
  blob = struct();
  for field = metadata.fieldnames(:)'
    field = field{1};
    sub_count = metadata.counts.(field);
    sub_vector = vector(count + (1:sub_count));
    sub_metadata = metadata.metadatas.(field);
    blob.(field) = VectorToBlob(sub_vector, sub_metadata);
    count = count + sub_count;
  end
  assert(count == length(vector));
else
  assert(false);
end
