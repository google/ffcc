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

function [loss, d_loss_model_vec, output] = ...
  TrainModelLossfunDeep(model_vec, data, regularizer, preconditioner, params)

compute_gradient = nargout >= 2;

model = VectorToBlob(model_vec, params.model_meta);

if compute_gradient
  d_loss_model = VectorToBlob(zeros(size(model_vec)), params.model_meta);
end

% If a different weight decay is specified for each layer, we can precondition
% optimization with those weight decay multipliers, and then simply impose a
% multiplier-free regularizer on the preconditioned weights.
if params.DEEP.PRECONDITION_WEIGHTS
  for d = 1:length(model.W)
    scale = 1 / sqrt(params.HYPERPARAMS.WEIGHT_DECAY(d));
    model.W{d} = model.W{d} * scale;
    model.b{d} = model.b{d} * scale;
  end
end

X = cat(2, data.feature_vec);

% Push the features through the neural network.
Ys = cell(1, length(model.W));
for d = 1:length(model.W)
  if d == 1
    Y_last = X;
  else
    Y_last = Ys{d-1};
  end
  Ys{d} = bsxfun(@plus, model.W{d} * Y_last, model.b{d});
  if d < length(model.W)
    Ys{d} = max(0, Ys{d});
  end
end

% Compute the loss at the final activation (the weights for AWB models).
loss = 0;
d_loss_Ys = cell(size(Ys));
d_loss_Ys{end} = zeros(size(Ys{end}));
for i_data = 1:length(data)
  y = Ys{end}(:,i_data);
  if compute_gradient
    [sub_loss, d_loss_y, ~, output] = TrainModelLossfun( ...
      y, data(i_data), regularizer, preconditioner, params);
    d_loss_Ys{end}(:,i_data) = d_loss_y;
  else
    sub_loss = TrainModelLossfun( ...
      y, data(i_data), regularizer, preconditioner, params);
  end
  loss = loss + sub_loss;
end

% Apply weight decay to each layer of the network.
for d = 1:length(model.W)
  if params.DEEP.PRECONDITION_WEIGHTS
    mult = length(data);
  else
    mult = params.HYPERPARAMS.WEIGHT_DECAY(d) * length(data);
  end

  loss = loss + (0.5 * mult) * sum(model.W{d}(:).^2);
  loss = loss + (0.5 * mult) * sum(model.b{d}(:).^2);
  if compute_gradient
    d_loss_model.W{d} = d_loss_model.W{d} + mult * model.W{d};
    d_loss_model.b{d} = d_loss_model.b{d} + mult * model.b{d};
  end
end

if compute_gradient
  for d = length(model.W):-1:1
    d_loss_model.b{d} = d_loss_model.b{d} + sum(d_loss_Ys{d}, 2);
    if d > 1
      d_loss_model.W{d} = d_loss_model.W{d} + d_loss_Ys{d} * Ys{d-1}';
      d_loss_Ys{d-1} = (model.W{d}' * d_loss_Ys{d}) .* (Ys{d-1} > 0);
    else
      d_loss_model.W{d} = d_loss_model.W{d} + d_loss_Ys{d} * X';
    end
  end

  % Backprop the gradient onto the preconditioned weights.
  if params.DEEP.PRECONDITION_WEIGHTS
    for d = 1:length(model.W)
      scale = 1 / sqrt(params.HYPERPARAMS.WEIGHT_DECAY(d));
      d_loss_model.W{d} = scale * d_loss_model.W{d};
      d_loss_model.b{d} = scale * d_loss_model.b{d};
    end
  end

  d_loss_model_vec = BlobToVector(d_loss_model);
end
