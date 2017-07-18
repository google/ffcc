% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

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
