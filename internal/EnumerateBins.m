function [bins, extents] = EnumerateBins(params)
% Takes as input a parameter struct, and produces a 1xNUM_BINS vector of the
% u/v value of each bin in the histogram implied by, STARTING_UV, NUM_BINS and
% BIN_SIZE.
% Also produces as optional output the extents of the histogram in UV
% coordinates (ie, the exact UV values at which the histogram "wraps around").

bins = params.HISTOGRAM.STARTING_UV + [0 : params.HISTOGRAM.BIN_SIZE ...
  : (params.HISTOGRAM.BIN_SIZE * (params.HISTOGRAM.NUM_BINS-1))];

extents = [bins(1) - params.HISTOGRAM.BIN_SIZE/2, ...
           bins(end) + params.HISTOGRAM.BIN_SIZE/2];
