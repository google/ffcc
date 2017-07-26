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

function BlobToVector_Test
% Checks that a flat struct of arrays with different sizes and types can be
% serialized to a vector and reconstructed to a struct correctly.

blob = struct(...
  'foo', double(randn(3,4,5)), ...
  'bar', single(randn(3,1)), ...
  'baz', uint8(255*rand(3,4,2)), ...
  'qux', int32((2^(32)-1)*randn(2,1,3,2)));

for field = fieldnames(blob)'
  field = field{1};
  mat = blob.(field);
  [vector, metadata] = BlobToVector(mat);
  mat_recon = VectorToBlob(vector, metadata);
  assert(all(size(mat_recon) == size(mat)));
  assert(all(mat_recon(:) == mat(:)));

  mat_recon = eval(BlobToString(mat));
    assert(max(max(max(max(abs(log(double(mat_recon)) ...
    - log(double(mat))))))) < 1e-5)
end

[blob_vec, metadata] = BlobToVector(blob);
blob_recon = VectorToBlob(blob_vec, metadata);

for field = fieldnames(blob)'
  field = field{1};
  assert(all(all(all(all(double(blob_recon.(field)) ...
    == double(blob.(field)))))));
  assert(strcmp(class(blob_recon.(field)), class(blob.(field))));
end

blob_recon = eval(BlobToString(blob));

for field = fieldnames(blob)'
  field = field{1};
  assert(max(max(max(max(abs(log(double(blob_recon.(field))) ...
    - log(double(blob.(field)))))))) < 1e-5)
end

fprintf('%s\n', BlobToString(blob));

clearvars;

% Checks that a single cell array of matrices can be serialized correctly.

blob = {randn(1, 2), 2, randn(3, 4); randn(1, 5), 3, 4};

[blob_vec, metadata] = BlobToVector(blob);
blob_recon = VectorToBlob(blob_vec, metadata);

for i = 1:numel(blob)
  assert(all(all(all(blob{i} == blob_recon{i}))));
end

blob_recon = eval(BlobToString(blob));

for i = 1:numel(blob)
  assert(max(max(max(max(abs(log(double(blob_recon{i})) ...
    - log(double(blob{i}))))))) < 1e-5)
end


fprintf('%s\n', BlobToString(blob));

clearvars

% Checks that a nested struct of structs can be serialized correctly.

blob = struct(...
  'foo', randn(3,4,5), ...
  'substruct1', struct(...
    'foo1', randn(2,1,4), ...
    'foo2', randn(1,2,3), ...
    'subsubstruct1', struct(...
    'subfoo1', randn(1,4,5), ...
    'subfoo2', randn(3,1,2))), ...
  'bar', randn(3,1), ...
  'substruct2', struct(...
    'bar1', randn(3,3), ...
    'bar2', randn(1,3)));

[blob_vec, metadata] = BlobToVector(blob);
blob_recon = VectorToBlob(blob_vec, metadata);

assert(all(all(all(blob.foo == blob_recon.foo))))
assert(all(all(all(blob.substruct1.foo1 == blob_recon.substruct1.foo1))));
assert(all(all(all(blob.substruct1.foo2 == blob_recon.substruct1.foo2))));
assert(all(all(all(blob.substruct1.subsubstruct1.subfoo1 == ...
  blob_recon.substruct1.subsubstruct1.subfoo1))));
assert(all(all(all(blob.substruct1.subsubstruct1.subfoo2 == ...
  blob_recon.substruct1.subsubstruct1.subfoo2))));
assert(all(all(all(blob.bar == blob_recon.bar))));
assert(all(all(all(blob.substruct2.bar1 == blob_recon.substruct2.bar1))));
assert(all(all(all(blob.substruct2.bar2 == blob_recon.substruct2.bar2))));

blob_recon = eval(BlobToString(blob));

assert(max(max(max(abs(blob.foo - blob_recon.foo)))) < 1e-5)
assert( ...
  max(max(max(abs(blob.substruct1.foo1 - blob_recon.substruct1.foo1)))) < 1e-5);
assert( ...
  max(max(max(abs(blob.substruct1.foo2 - blob_recon.substruct1.foo2)))) < 1e-5);
assert(max(max(max(abs(blob.substruct1.subsubstruct1.subfoo1 - ...
  blob_recon.substruct1.subsubstruct1.subfoo1)))) < 1e-5);
assert(max(max(max(abs(blob.substruct1.subsubstruct1.subfoo2 - ...
  blob_recon.substruct1.subsubstruct1.subfoo2)))) < 1e-5);
assert(max(max(max(abs(blob.bar - blob_recon.bar)))) < 1e-5);
assert( ...
  max(max(max(abs(blob.substruct2.bar1 - blob_recon.substruct2.bar1)))) < 1e-5);
assert( ...
  max(max(max(abs(blob.substruct2.bar2 - blob_recon.substruct2.bar2)))) < 1e-5);


fprintf('%s\n', BlobToString(blob));

clearvars;

% Checks that a nested cell array of structs and arrays can be serialized
% correctly.

blob1 = struct(...
  'foo', double(randn(3,4,5)), ...
  'bar', uint8(255*rand(2,4,3)));

blob2 = struct(...
  'baz', double(randn(3,4,5)), ...
  'qux', uint8(255*rand(3,4,2)));

blob = {blob1, blob2, randn(2, 3)};

[blob_vec, metadata] = BlobToVector(blob);
blob_recon = VectorToBlob(blob_vec, metadata);

assert(all(all(all(blob{1}.foo == blob_recon{1}.foo))));
assert(all(all(all(blob{1}.bar == blob_recon{1}.bar))));
assert(all(all(all(blob{2}.baz == blob_recon{2}.baz))));
assert(all(all(all(blob{2}.qux == blob_recon{2}.qux))));
assert(all(all(all(blob{3} == blob_recon{3}))));

blob_recon = eval(BlobToString(blob));

assert(max(max(max(abs(double(blob{1}.foo) - double(blob_recon{1}.foo))))) < 1e-5);
assert(max(max(max(abs(double(blob{1}.bar) - double(blob_recon{1}.bar))))) < 1e-5);
assert(max(max(max(abs(double(blob{2}.baz) - double(blob_recon{2}.baz))))) < 1e-5);
assert(max(max(max(abs(double(blob{2}.qux) - double(blob_recon{2}.qux))))) < 1e-5);
assert(max(max(max(abs(double(blob{3}) - double(blob_recon{3}))))) < 1e-5);

fprintf('%s\n', BlobToString(blob));

fprintf('Tests Passed\n');
