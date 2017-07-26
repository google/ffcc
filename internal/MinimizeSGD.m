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

function state = MinimizeSGD(options, lossfun, state, data, varargin)
% An implementation of the ADAM stochastic gradient descent algorithm (Kingma
% and Ba 2014), but with the parameters defined in terms of half-life instead
% of exponential decay values, and with the step size parametrized as an initial
% value and final value, where we take the weighted lerp of the log of the two
% as a function of epoch. Takes as input:
%   options  - A struct with parameters that determine how optimization behaves.
%              Contains the following fields:
%       num_epochs      - The number of passes over the data.
%       init_step_size  - The initial step size of each update.
%       final_step_size - The final step size of each update.
%       grad_halflife   - The half-life for the gradient estimate.
%       hess_halflife   - The half-life for the hessian diagonal estimate.
%       hess_epsilon    - The padding on the denominator during normalization.
%       display         - If true, print information during optimization.
%   lossfun  - The loss being minimized, which is a function handle of the form
%              [loss, d_loss] = lossfun(state, data, varargin{:})
%   state    - A vector describing the model.
%   data     - A vector of structs, each element of which is a datapoint.
%   varargin - Whatever other arguments need to be passed to the loss function.

% A function handle for converting a half life (the number of steps after which
% some value should have decayed by half) into an exponential decay value.
halflife2rho = @(x)(2.^-(1./x));

loss_rho = halflife2rho(length(data)); % loss half-life = # datapoints.
grad_rho = halflife2rho(options.grad_halflife);
hess_rho = halflife2rho(options.hess_halflife);

if options.display
  fprintf('grad_rho = %e\n', grad_rho);
  fprintf('hess_rho = %e\n', hess_rho);
  fprintf('hess_epsilon = %e\n', options.hess_epsilon);
  fprintf('init_step_size = %e\n', options.init_step_size);
  fprintf('final_step_size = %e\n', options.final_step_size);
  fprintf('Iteration    EpochLoss     SmoothLoss     StepSize    Sec./Epoch\n');
end

loss_numer = 0;
loss_denom = 0;
grad_numer = zeros(size(state));
grad_denom = 0;
hess_numer = zeros(size(state));
hess_denom = 0;

rng('default')

% Pass over all of the data num_epochs times.
for epoch = 1:options.num_epochs

  tic;
  loss_sum = 0;

  % Iterate through each datapoint individually, in a random order.
  data_order = randperm(length(data));
  for iter = 1:length(data_order)
    i_data = data_order(iter);

    t = ((epoch-1) * length(data_order) + iter - 1) ...
      ./ (options.num_epochs * length(data_order)-1);
    step_size = exp(log(options.init_step_size) * (1-t) ...
                  + log(options.final_step_size) * t);

    % Get the loss and its gradient.
    [sub_loss, sub_d_loss] = lossfun(state, data(i_data), varargin{:});

    % Update our running estimate of loss, the gradient, and the
    % diagonal of the hessian (the gradient squared).
    loss_numer = loss_rho * loss_numer + (1-loss_rho) * sub_loss;
    loss_denom = loss_rho * loss_denom + (1-loss_rho);

    grad_numer = grad_rho * grad_numer + (1-grad_rho) * sub_d_loss;
    grad_denom = grad_rho * grad_denom + (1-grad_rho);

    hess_numer = hess_rho * hess_numer + (1-hess_rho) * sub_d_loss.^2;
    hess_denom = hess_rho * hess_denom + (1-hess_rho);

    % Normalize the current gradient estimate by the current hessian estimate
    % and use it as the update to the state.
    grad_batch = grad_numer ./ grad_denom;
    hess_batch = sqrt((hess_numer ./ hess_denom) + options.hess_epsilon^2);
    update_batch = grad_batch ./ hess_batch;
    state = state - step_size * update_batch;

    % Record the sum of the losses for each datapoint.
    loss_sum = loss_sum + sub_loss;
  end

  % Compute the smooth running average of the losses for all data.
  loss_avg = loss_numer ./ loss_denom * length(data);

  if options.display
    fprintf('%9d    %+0.3e    %+0.3e     2^%+0.3f    %0.3fs', ...
            epoch, loss_sum, loss_avg, log2(step_size), toc);
    fprintf('\n');
  end

end
