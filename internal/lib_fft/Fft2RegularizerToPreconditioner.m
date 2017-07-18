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

function V_pre = Fft2RegularizerToPreconditioner(F_reg)
% Given a regularizer in the frequency domain (a per-FFT-bin non-negative
% value indicating how much that frequency will be regularized) we construct a
% Jacobi preconditioner, which is the the inverse diagonal of that
% quadratic regularizer.
% "Preconditioning" is a bit of a misnomer here, as we are actually
% "pretransforming" our model, and performing LBFGS in this transformed space.
% As a result, the "preconditioner" scalings that we produce are actually the
% square-root of what the actual preconditioner should be, as these scalings are
% effectively applied twice.

assert(all(F_reg(:) > 0));  % Prevent divide by zero.

% Because of the sqrt(2) scaling in Fft2ToVec
V_pre = sqrt(sqrt(2) ./ Fft2ToVec(complex(F_reg, F_reg)));

% We need to special-case the 4 FFT elements that are present only once.
n = size(F_reg,1);
scale_idx = [1; n/2 + 1; n/2 + 2; n + 2];
V_pre(scale_idx,:) = V_pre(scale_idx,:) ./ sqrt(sqrt(2));
