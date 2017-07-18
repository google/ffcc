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

function TuneParameters_Test
% Tests that TuneParameters is behaving correctly, by ensuring that it
% finds the the minimum of a simple function within a fixed number of
% iterations.

clearvars;

rng('default')
num_iters = [];
for iter = 1:10
  % CrossValidate supports a mock error function for debugging
  % TuneParameters(), which we can enable here.
  error_function = @(x)(x.rgb_err.mean);

  params = PrivateConstants;
  params.TUNING.DEBUG_MODE = true;

  % Initialize the hyperparameters randomly in quadrant 1.
  hyperparams = struct('x', 2.^(4*rand-2), 'y', 2.^(4*rand-2));
  [params, history] = TuneParameters([], params, [], error_function, ...
    hyperparams, params.TUNING, '');

  try
    [log_x, log_y] = ndgrid(-4:(1/16):4, -4:(1/16):4);
    vis = reshape(max(abs([log_x(:), log_y(:)]), [], 2), size(log_x));

    % Render the trace of parameter tuning, with color proportional to order.
    cmap = colormap('jet');
    color_idx = ...
      round(1+[0:(size(history.hyperparams,2)-1)]' ...
      / (size(history.hyperparams,2)-1) * (size(cmap,1)-1));
    imagesc(log_x(:,1), log_x(:,1), vis); hold on;
    scatter(log(history.hyperparams(1,:)), ...
            log(history.hyperparams(2,:)), ...
            40, cmap(color_idx(:,:),:), 'filled');
    axis square;
    axis([-4, 4, -4, 4]);
    drawnow
    hold off;
  catch me
    me;
  end

  % Because our loss function is known, we can require that our distance to
  % the optimal hyperparameter point (1,1) is within some function of the
  % search multiplier.
  assert(min(history.errors) <= log(params.TUNING.MIN_MULT))
  num_iters(iter) = length(history.errors);
end

% These are brittle 'golden' tests to check against regressions in
% TuneParameters() that might cause it to start taking a large number of
% unproductive iterations.
ErrorMetrics(num_iters)
assert(mean(num_iters) < 22)
assert(max(num_iters) <= 27)

try
  hist(num_iters, min(num_iters) : max(num_iters))
catch me
  me;
end

fprintf('Test Passed\n');
