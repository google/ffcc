function [loss_rho, varargout] = ApplyPower(rho, loss, varargin)
% Given some power rho, raises the loss to that power. The variable length
% additional arguments are assumed to be partial derivatives with respect
% to the loss, and are modified accordingly.

% This operation is not specified for negative values.
assert(loss >= 0);

loss_rho = loss.^rho;

% D[f[x]^rho, x] = rho * f[x]^(rho-1) * f'[x]
% Every element in varargin is assumed to be some f'[x].
grad = rho * loss.^(rho-1);
for i = 1:length(varargin)
  varargout{i} = grad * varargin{i};
end
