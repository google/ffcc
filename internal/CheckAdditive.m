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
