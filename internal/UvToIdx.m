function Y_idx = UvToIdx(Y, params)
% Maps from a 2xn matrix of UV values to their nearest corresponding
% (possibly-aliased) histogram index.

assert(size(Y,1) == 2);

wrap = @(x)(mod(x, params.HISTOGRAM.NUM_BINS)+1);
Y_idx = wrap( ...
  round((Y - params.HISTOGRAM.STARTING_UV) / params.HISTOGRAM.BIN_SIZE));
