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

function CheckAdditive(lossfun, model, data, num_checks, args)
% This test checks that the loss of a chunk of data is equal to the sum
% of the losses of any two partitions of that data, by randomly
% generating models and partitions of data.

[model_vec, params.model_meta] = BlobToVector(model);

is_additive = nan(num_checks,1);
for i_rand = 1:num_checks
  model_rand = randn(size(model_vec))/10;
  idx = floor(rand * length(data));
  loss1 = lossfun(model_rand, data(1:idx), args{:});
  loss2 = lossfun(model_rand, data((idx+1):end), args{:});
  loss_sum = lossfun(model_rand, data, args{:});
  additive_loss_error = abs(loss1 + loss2 - loss_sum);
  is_additive(i_rand) = (additive_loss_error < 1e-8);
  if is_additive(i_rand)
    fprintf('1');
  else
    fprintf('0');
  end
  if (mod(i_rand, 80) == 0) || (i_rand == num_checks)
    fprintf('\n');
  end
end
fprintf('%.2f%% additive\n', 100*mean(is_additive))
