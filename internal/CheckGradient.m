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

function CheckGradient(lossfun, model, ...
  step_size, model_sigma, num_fast_checks, num_dims, args)
% Runs a series of checks on loss function defined by lossfun, using the
% input model as a template and a starting point.
%   step_size - The size of finite-difference checks
%   model_sigma - The standard deviation of the white noise injected into
%                 the input model to make random model variants.
%   num_fast_checks - The number of fast whole-model gradient checks to make
%                     before doing the per-dimension gradient checks.
%   num_dims - The number of dimensions of the model to check with finite
%              differences.

[model_vec, params.model_meta] = BlobToVector(model);
model_vec_sz = size(model_vec);

% Test the complete gradient by looking at random vectors which are added
% to the model.
for i_trial = 1:num_fast_checks
  fastDerivativeCheck(lossfun, ...
    model_vec + randn(model_vec_sz) * model_sigma, 1, 1, args{:});
end

% Pick a random point nearby the input model (the input model is likely at a
% "hinge" in the loss function).
model_vec = model_vec + randn(model_vec_sz) * model_sigma;

% Test a subset of gradient by varying dimensions independently (ie,
% compute a subset of the gradient using finite differences).
[loss, d_loss] = lossfun(model_vec, args{:});

n_loss = nan(size(d_loss));
ds = 1:round(prod(model_vec_sz)/num_dims):prod(model_vec_sz);
for di = 1:length(ds)
  PrintDots(di, length(ds));
  d = ds(di);
  model_vec_step = model_vec;
  model_vec_step(d) = model_vec_step(d) + step_size;
  loss_step = lossfun(model_vec_step, args{:});
  n_loss(d) = (loss_step - loss) / step_size;
end

% Display stats of the difference between the analytical gradient and the
% numerical gradient.
fprintf('Per-Dimension Gradient Errors:\n');
valid = ~isnan(n_loss);
percentiles = [50, 90, 99, 100];
absolute_errors = prctile(abs(n_loss(valid) - d_loss(valid)), percentiles);
relative_errors = prctile(abs(n_loss(valid) - d_loss(valid)) ./ ...
  max(eps, abs(d_loss(valid))), percentiles);
fprintf('Absolute Gradient Errors:\n');
fprintf('%3d%% = %g\n', [percentiles; absolute_errors])
fprintf('\n');
fprintf('Relative Gradient Errors:\n');
fprintf('%3d%% = %g\n', [percentiles; relative_errors])
fprintf('\n');

% Render the two gradients, if possible.
try
  plot(n_loss(~isnan(n_loss)), 'bo'); hold on;
  plot(d_loss(~isnan(n_loss)), 'rx'); hold off;
  drawnow;
catch me
  fprintf('%s - %s\n', me.identifier, me.message)
end
