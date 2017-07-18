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
