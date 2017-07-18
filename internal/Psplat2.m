function N = Psplat2(u, v, c, bin_lo, bin_step, n_bins)
% Splat a bunch of (u, v) coordinates to a 2D histogram. We splat with a
% periodic edge condition, which is important because we will later
% convolve with periodic edges (FFT)

ub = 1 + mod(round((u - bin_lo) / bin_step), n_bins);
vb = 1 + mod(round((v - bin_lo) / bin_step), n_bins);
N = reshape(accumarray(sub2ind([n_bins, n_bins], ub(:), vb(:)), c(:), ...
                               [n_bins^2, 1]), n_bins, n_bins);
