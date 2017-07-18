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
