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

function KalmanUpdate_2D_Test
rng('default')

kNumDatapoints = 100;
kNumParticles = 100000;
sigma_scale = 128;

X_init = [0; 0];
V_init = 100 * eye(2);
V_transition = .003 * eye(2);

log_Sigma = randn(2,2);
log_Sigma = (log_Sigma + log_Sigma')/2;
Sigma = expm(log_Sigma) / sigma_scale;
log_Sigma = logm(Sigma);

X_true = {};
V_true = {};
for t = 1:kNumDatapoints
  X_true{t} = [sin((t)/16); cos((t)/8)];
  r = randn(2,2);
  r = (r + r')/2;
  r = logm(expm(r)/sigma_scale);

  log_Sigma = 0.5 * log_Sigma + 0.5 * r;
  V_true{t} = expm(log_Sigma);
end

X_obs = {};
for t = 1:kNumDatapoints
  X_obs{t} = mvnrnd(X_true{t}, V_true{t})';
end

X_particle = X_init;
V_particle = V_init;

X_particle_history = {};
V_particle_history = {};
X_kalman_history = {};
V_kalman_history = {};
for t = 1:kNumDatapoints

  particles = mvnrnd(X_particle, V_particle + V_transition, kNumParticles)';
  prob = mvnpdf(particles', X_obs{t}', V_true{t})';
  prob = prob ./ sum(prob);
  X_particle = sum(bsxfun(@times, prob, particles),2);

  particles_centered = bsxfun(@minus, particles, X_particle);
  V_particle = particles_centered * bsxfun(@times, prob, particles_centered)';

  if t == 1
    state_last = [];
  else
    state_last = struct('mu', X_kalman, 'Sigma', V_kalman);
  end
  state_obs = struct('mu', X_obs{t}, 'Sigma', V_true{t});
  state_next = KalmanUpdate(state_obs, state_last, V_transition);
  X_kalman = state_next.mu;
  V_kalman = state_next.Sigma;
  X_kalman_history{t} = X_kalman;
  V_kalman_history{t} = V_kalman;

  X_particle_history{t} = X_particle;
  V_particle_history{t} = V_particle;

  try
    scatter(X_true{t}(1), X_true{t}(2), 'ko'); hold on;
    line([X_true{t}(1), X_obs{t}(1)], [X_true{t}(2), X_obs{t}(2)], ...
      'Color', 'k');
    scatter(X_particle(1), X_particle(2), 'b+');
    scatter(X_kalman(1), X_kalman(2), 'rx'); hold off;
    set(gca, 'XLim', 1.2 * [-1,1])
    set(gca, 'YLim', 1.2 * [-1,1])
    axis square;
    drawnow;
  catch me
  end
end

true_MSE = sum(mean((cat(2, X_particle_history{:}) - cat(2, X_true{:})).^2,2));
fprintf('Kalman filter absolute MSE: %e\n', true_MSE);

particle_MSE = sum(mean((cat(2, X_particle_history{:}) ...
                       - cat(2, X_kalman_history{:})).^2,2));
fprintf('Kalman filter relative MSE to particle filter: %e\n', particle_MSE);

assert(true_MSE < 1e-1)
assert(particle_MSE < 1e-4)

fprintf('Test Passed\n');
