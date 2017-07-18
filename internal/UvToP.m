function P = UvToP(y, params)
% Maps from a 2x1 vector of UV values to a bilinearly interpolated PDF.

assert(size(y,1) == 2);
assert(size(y,2) == 1);

y_ij = (y - params.HISTOGRAM.STARTING_UV) / params.HISTOGRAM.BIN_SIZE;

y_lo = floor(y_ij);
y_hi = y_lo + 1;

w_1 = y_ij - y_lo;
w_0 = 1 - w_1;
w_00 = w_0(1) * w_0(2);
w_01 = w_0(1) * w_1(2);
w_10 = w_1(1) * w_0(2);
w_11 = w_1(1) * w_1(2);

wrap = @(x)(mod(x, params.HISTOGRAM.NUM_BINS)+1);

P = sparse( ...
  [wrap(y_lo(1)); wrap(y_lo(1)); wrap(y_hi(1)); wrap(y_hi(1))], ...
  [wrap(y_lo(2)); wrap(y_hi(2)); wrap(y_lo(2)); wrap(y_hi(2))], ...
  [w_00; w_01; w_10; w_11], ...
  params.HISTOGRAM.NUM_BINS, params.HISTOGRAM.NUM_BINS);
