function dF = Fft2ToVecBackprop(dV, varargin)
% Backpropagates through Fft2ToVec().

dF = 2 * VecToFft2(dV, varargin{:});
