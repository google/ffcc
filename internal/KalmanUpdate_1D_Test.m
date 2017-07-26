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

function KalmanUpdate_1D_Test
rng('default')

kNumDatapoints = 100;
kNumParticles = 10000;
X_init = 0;
V_init = 100;
V_transition = .01;

X_true = sin((1:kNumDatapoints)/10);
V_true = exp((randn(size(X_true))-5));

X_obs = X_true + sqrt(V_true) .* randn(size(X_true));

X_kalman = X_init;
V_kalman = V_init;

X_particle = X_init;
V_particle = V_init;

X_particle_history = nan(size(X_obs));
V_particle_history = nan(size(X_obs));
X_kalman_history = nan(size(X_obs));
V_kalman_history = nan(size(X_obs));
for t = 1:kNumDatapoints
  particles = ...
    X_particle + sqrt(V_particle+ V_transition) * randn(kNumParticles,1);
  prob = normpdf(particles, X_obs(t), sqrt(V_true(t)));
  prob = prob ./ sum(prob);
  X_particle = sum(prob .* particles);
  V_particle = sum(prob .* particles.^2) - X_particle.^2;

  X_particle_history(t) = X_particle;
  V_particle_history(t) = V_particle;

  if t == 1
    state_last = [];
  else
    state_last = struct('mu', X_kalman, 'Sigma', V_kalman);
  end
  state_obs = struct('mu', X_obs(t), 'Sigma', V_true(t));
  state_next = KalmanUpdate(state_obs, state_last, V_transition);
  X_kalman = state_next.mu;
  V_kalman = state_next.Sigma;

  X_kalman_history(t) = X_kalman;
  V_kalman_history(t) = V_kalman;
end

true_MSE = mean((X_kalman_history - X_true).^2);
particle_MSE = mean((X_particle_history - X_kalman_history).^2);

fprintf('Kalman filter absolute MSE: %e\n', true_MSE);
fprintf('Kalman filter relative MSE to particle filter: %e\n', particle_MSE);

assert(true_MSE < 1e-2)
assert(particle_MSE < 1e-5)

fprintf('Test Passed\n');

try
  is = -1.5:0.01:1.5;
  js = 1:0.1:kNumDatapoints;
  [xx,ii] = ndgrid(is, js);
  V_up = 1./interp1(1./V_kalman_history, ii(:), 'linear');
  X_up = interp1(X_kalman_history ./ V_kalman_history, ii(:), 'linear') .* V_up;
  P = reshape(normpdf(xx(:), X_up, sqrt(V_up)), size(xx));

  imagesc(js, is, P); colormap('hot'); hold on;
  plot(X_true, 'w-');
  for t = 1:kNumDatapoints
    plot([t, t], X_obs(t) + sqrt(V_true(t)) * [-1,1], 'y-');
  end
  plot(X_kalman_history, 'c-');
  plot(X_kalman_history + sqrt(V_kalman_history), 'c:');
  plot(X_kalman_history - sqrt(V_kalman_history), 'c:');
  set(gca, 'YLim', [-1.5, 1.5]);
  axis off square;
  drawnow;
  hold off;
catch me
  me;
end
