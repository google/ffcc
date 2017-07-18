function [uv, Sigma_uv] = IdxToUv(idx, Sigma_idx, params)
% Turn the one-indexed (i,j) coordinate into a (u,v) chroma coordinate.
% This function does no periodic "unwrapping".
% Can be called with three parameters in which case the second parameter is
% assumed to be a covariance matrix, or two parameters in which case the
% function behaves as:
%   uv = IdxToUV(idx, params)

assert((nargin == 2) || (nargin == 3))
if nargin == 2
  params = Sigma_idx;
  Sigma_idx = [];
end

uv = (idx - 1) * params.HISTOGRAM.BIN_SIZE + params.HISTOGRAM.STARTING_UV;
if ~isempty(Sigma_idx)
  Sigma_uv = Sigma_idx * (params.HISTOGRAM.BIN_SIZE.^2);
end
