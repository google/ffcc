function ComputeWhiteningTransformation_Test

rng(0)

d = 4;
n = 100000;

% Construct a random PSD affine transformation.
r = randn(d, d);
A = expm((r + r')/2);
mu = randn(d, 1);

% Sample white data.
X_white = randn(d, n);

% Transform the data.
X = bsxfun(@plus, A * X_white, mu);

% Compute a whitening transformation
T = ComputeWhiteningTransformation(X, true, 0, 'zca');

% Check that the transformation is correct.
assert(max(max(abs(T.inv_A - A))) < 0.01)
assert(max(abs(T.mu - mu)) < 0.01)
assert(max(max(abs(T.whiten(X) - X_white))) < 0.1)
assert(max(max(abs(T.unwhiten(X_white) - X))) < 0.1);

fprintf('Tests Passed\n');
