function CheckConvex(lossfun, model, data, num_checks, num_data, args)
% If the loss function f(x) is convex, then it must be that
%   f( (x_1 + x_2) / 2 ) <= (f(x_1) + f(x_2)) / 2
% This means that we can check the convexity of the loss by repeatedly
% generating pairs of random model weights, and asserting the above
% inequality.

[model_vec, params.model_meta] = BlobToVector(model);

rng('default')
randn_sigmas = exp(2*randn(num_checks,1));
is_convex = nan(num_checks, 1);
for iter = 1:num_checks
  idx = randperm(length(data));
  idx = idx(1:min(length(idx), num_data));
  data_sub = data(idx);

  randn_sigma = randn_sigmas(iter);
  model1_vec = randn(size(model_vec)) * randn_sigma;
  model2_vec = randn(size(model_vec)) * randn_sigma;
  modelH_vec = (model1_vec + model2_vec)/2;
  model1 = VectorToBlob(model1_vec, params.model_meta);
  model2 = VectorToBlob(model2_vec, params.model_meta);
  modelH = VectorToBlob(modelH_vec, params.model_meta);

  loss = @(x)(lossfun(BlobToVector(x), data_sub, args{:}));

  loss1 = loss(model1);
  loss2 = loss(model2);
  lossH = loss(modelH);
  is_convex(iter) = lossH <= ((loss1 + loss2)/2 + 1e-5);

  if is_convex(iter)
    fprintf('1')
  else
    fprintf('0')
  end
  if (mod(iter, 80) == 0) || (iter == num_checks)
    fprintf('\n');
  end
end
fprintf('%.2f%% convex\n', 100*mean(is_convex))
