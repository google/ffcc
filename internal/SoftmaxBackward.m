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
