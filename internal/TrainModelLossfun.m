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

function [loss, d_loss_precond_vec, model, output] = TrainModelLossfun(...
  model_precond_vec, data, regularizer, preconditioner, params)

% Unpack the model from a vector to a struct.
model_precond = VectorToBlob(model_precond_vec, params.shallow_model_meta);

model_fields = {'F'};

if params.TRAINING.LEARN_BIAS
  model_fields{end+1} = 'B';
end

for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft = [Q, '_fft'];
  Q_fft_latent = [Q, '_fft_latent'];

  model.(Q_fft_latent) = bsxfun(@times, preconditioner.(Q_fft_latent), ...
    model_precond.(Q_fft_latent));
  model.(Q_fft) = VecToFft2(model.(Q_fft_latent), params.fft_mapping);
  model.(Q) = ifft2(model.(Q_fft));
end

loss = 0;
d_loss.F_fft = zeros(size(model.F_fft));
if params.TRAINING.LEARN_BIAS
  d_loss.B = zeros(size(model.B));
end
data_mass = 0;  % The total weight of all datapoints.

Y_pred = {};
for i_data = 1:length(data)

  X = data(i_data).X;

  if ~isfield(data(i_data), 'X_fft')
    X_fft = fft2(X);
  else
    X_fft = data(i_data).X_fft;
  end
  Y = data(i_data).Y;
  avg_rgb = data(i_data).avg_rgb;

  F_fft = model.F_fft;
  if params.TRAINING.LEARN_BIAS
    B = model.B;
  else
    B = [];
  end

  [output1, output2, sub_loss, output4] ...
    = EvaluateModel(F_fft, B, X, X_fft, Y, [], avg_rgb, params);
  output = {output1, output2, sub_loss, output4};

  Y_pred{i_data} = output1.mu;

  % Each datapoint is weighted, but that weight is currently fixed to 1.
  W = 1;

  % Add to the total loss and its gradients.
  data_mass = data_mass + W;
  loss = loss + W * sub_loss.loss;
  d_loss.F_fft = d_loss.F_fft + W * sub_loss.d_loss_F_fft;
  if params.TRAINING.LEARN_BIAS
    d_loss.B = d_loss.B + W * sub_loss.d_loss_B;
  end
end
Y_pred = cat(2, Y_pred{:});

if params.TRAINING.LEARN_BIAS
  d_loss.B_fft = (1 / (size(model.B,1) * size(model.B,2))) * fft2(d_loss.B);
end

% Vectorize and precondition the gradients.
for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft = [Q, '_fft'];
  Q_fft_latent = [Q, '_fft_latent'];
  d_loss.(Q_fft_latent) = VecToFft2Backprop(d_loss.(Q_fft));
end

d_loss_precond = struct();
for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft_latent = [Q, '_fft_latent'];
  d_loss_precond.(Q_fft_latent) = ...
    bsxfun(@times, preconditioner.(Q_fft_latent), d_loss.(Q_fft_latent));
end

% Regularize each model parameter, in the preconditioned space. The magnitude of
% each regularizer is multiplied by the total weight of all datapoints
% (which makes the loss additive). Because this loss is computed and imposed in
% the vectorized and preconditioned space, it is just the sum of squares.
for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft = [Q, '_fft'];
  Q_fft_latent = [Q, '_fft_latent'];

  loss = loss + ...
    0.5 * data_mass * sum(reshape(model_precond.(Q_fft_latent), [], 1).^2);
  d_loss_precond.(Q_fft_latent) = d_loss_precond.(Q_fft_latent) + ...
    data_mass * model_precond.(Q_fft_latent);

  if params.DEBUG.CORRECT_PRECONDITIONER
    error = ...
      (sum(model_precond.(Q_fft_latent)(:).^2)) -  ...
      sum(sum(sum(bsxfun(@times, regularizer.(Q_fft), ...
      (real(model.(Q_fft)).^2 + imag(model.(Q_fft)).^2)))));
    fprintf('========================================\n');
    fprintf('%s preconditioner error    = %e\n', Q, abs(error));
  end
end

d_loss_precond_vec = BlobToVector(d_loss_precond);

if params.DEBUG.RENDER_MODEL

  % Initialize a global clock.
  global display_last_time
  if isempty(display_last_time)
    display_last_time = clock;
  end

  % If the amount of time that has passed since the last rendering of the model
  % exceeds RENDER_PERIOD, render the model and reset the clock.
  if etime(clock, display_last_time) > params.DEBUG.RENDER_PERIOD
    display_last_time = clock;

    try
      figure(1);

      % Invert the Fourier-domain filter bank.
      F = {};
      for c = 1:size(model.F_fft,3)
        F{c} = fftshift(ifft2(model.F_fft(:,:,c)));
      end
      F = cat(3, F{:});

      F_range = [min(F(:)), max(F(:))];

      % Center an normalize the filter bank.
      F_centered = bsxfun(@minus, F, mean(mean(F,2),1));
      vis_F = bsxfun(@rdivide, F_centered, ...
        max(eps, max(max(abs(F_centered), [], 1), [], 2)));
      vis_F = reshape(vis_F, size(vis_F,1), []);

      % Shift and normalize the FFT filter bank.
      vis_F_fft = cell(1, size(model.F_fft,3));
      for c = 1:size(model.F_fft,3)
        vis_F_fft{c} = fftshift(abs(fft2(F_centered(:,:,c))));
      end
      vis_F_fft = cat(3, vis_F_fft{:});
      vis_F_fft = bsxfun(@rdivide, vis_F_fft, ...
        max(eps, max(max(vis_F_fft, [], 1), [], 2)));
      vis_F_fft = reshape(vis_F_fft, size(vis_F_fft,1), []);

      if params.TRAINING.LEARN_BIAS
        B = model.B;
        B_range = [inf, -inf];
        B_range(1) = min(B_range(1), min(B(:)));
        B_range(2) = max(B_range(2), max(B(:)));
        B_range(1) = B_range(1) - 10*eps;
        B_range(2) = B_range(2) + 10*eps;

        % Center and normalize the gain maps.
        vis_blackbody = [ ...
          (model.B - B_range(1)) / (B_range(2) - B_range(1)); ...
          zeros(size(model.B))];
      else
        vis_blackbody = [];
        B_range = [nan, nan];
      end

      % Render the model.
      vis = [[vis_F; vis_F_fft], vis_blackbody];
      imagesc(vis, [-1, 1]);
      ColormapUsa;
      axis image off;
      title(sprintf('F in [%0.3f, %0.3f], B in [%0.3f, %0.3f]\n', ...
        F_range(1), F_range(2), B_range(1), B_range(2)));

      % Render a quiver plot of the training data relative to our current
      % predictions.
      figure(2);
      Y_true = cat(2, data.Y);
      Y_delta = Y_pred - Y_true;
      Y_err = sum(Y_delta.^2, 1);
      keep = Y_err > prctile(Y_err, 90);  % Only render the worst 10%.
      quiver( ...
        Y_true(1,keep), Y_true(2,keep), Y_delta(1,keep), Y_delta(2,keep), ...
        0, ...  % Forces the quiver vectors to not be auto-resized.
        'Marker', 'o', ...
        'MarkerEdgeColor', 'k', ...
        'MarkerFaceColor', 'k', ...
        'MarkerSize', 3, ...
        'MaxHeadSize', 0);
      bins = EnumerateBins(params);
      axis([bins(1), bins(end), bins(1), bins(end)]);
      axis square ij;
      drawnow;

    catch me
      fprintf('%s\n', me.message)
    end
  end
end
