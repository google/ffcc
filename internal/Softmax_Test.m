function Softmax_Test
% Tests that Softmax is behaving and backprop'ing correctly.
clearvars;
rng('default')

% Check the simple 2D case where are dimensions are softmax'ed
H = exp(randn(32,32));
[P, meta] = SoftmaxForward(H);

% P should sum to 1.
assert(abs(sum(P(:)) - 1) < 1e-12);

% SoftmaxForward should be shift-invariant.
for i_rand = 1:100
  H2 = H + randn*100;
  P2 = SoftmaxForward(H2);
  assert(max(abs(P2(:) - P(:))) < 1e-12)
end

lossfun = @(H)(0.5 * sum((H(:) - 1).^2));
d_lossfun = @(H)(H - 1);

loss = lossfun(P);
dP = d_lossfun(P);

dH = SoftmaxBackward(dP, meta);

ep = 1e-5;
nH = nan(size(dH));
for d = 1:numel(H);
  H_ = H;
  H_(d) = H_(d) + ep;
  P_ = SoftmaxForward(H_);
  loss_ = lossfun(P_);
  nH(d) = (loss_ - loss) / ep;
end

assert(max(abs(nH(:) - dH(:))) < 1e-4);

% Check the support for arbitrary dimensions for a high-dimensional tensor.

H = exp(randn(3,4,5,6));
dims = [2, 4];
[P, meta] = SoftmaxForward(H, dims);

% P should sum to 1.
assert(all(all(abs(sum(sum(P, dims(1)), dims(2)) - 1) < 1e-12)))

% SoftmaxForward should be shift-invariant.
for i_rand = 1:100
  H2 = H + randn*100;
  P2 = SoftmaxForward(H2, dims);
  assert(max(abs(P2(:) - P(:))) < 1e-12)
end

lossfun = @(H)(0.5 * sum((H(:) - 1).^2));
d_lossfun = @(H)(H - 1);

loss = lossfun(P);
dP = d_lossfun(P);

dH = SoftmaxBackward(dP, meta);

ep = 1e-5;
nH = nan(size(dH));
for d = 1:numel(H);
  H_ = H;
  H_(d) = H_(d) + ep;
  P_ = SoftmaxForward(H_, dims);
  loss_ = lossfun(P_);
  nH(d) = (loss_ - loss) / ep;
end

assert(max(abs(nH(:) - dH(:))) < 1e-4);

fprintf('Test Passed\n');
