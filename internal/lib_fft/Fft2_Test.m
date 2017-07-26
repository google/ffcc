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

function Fft2_Test
rng('default')

% Iterate over sizes (powers of 2)
for n = 2.^[2:7]
  fprintf('n = %d\n', n);

  % Precompute a fast mapping.
  mapping = Fft2ToVecPrecompute([n, n]);

  % Iterate over outer dimension sizes.
  for c = 1:3
    for d = 1:3
      % Construct a random image and its FFT.
      X = randn(n, n, c, d);
      F = fft2(X);

      % Sanity check that the fft preserves magnitude.
      assert(all(abs( ...
        reshape(sum(sum(X.^2,1),2), [], 1) ...
        - reshape(mean(mean(abs(F).^2,2),1), [], 1)) < 1e-8));

      % Vectorize and unvectorize F using the normal code and the fast
      % mapping.
      for use_mapping = [false, true]
        if use_mapping
          V = Fft2ToVec(F, mapping);
        else
          V = Fft2ToVec(F);
        end

        % Check that the sizes all make sense.
        assert(length(V) == (n*n));
        % This test may be flaky, but it is extremely unlikely.
        for i = 1:size(V,2)
          assert(length(unique(V(:,i))) == (n*n))
        end
        assert(size(V,2) == size(F,3))

        % Unvectorize the fft and check that it is correct.
        F_ = VecToFft2(V);
        assert(all(size(F_) == size(F)));
        assert(max(abs(F_(:) - F(:))) < 1e-8)
      end

      % Check that vectorization preserves magnitude
      assert(all(abs( ...
        reshape(mean(V.^2,1), [], 1) ...
        - reshape(mean(mean(abs(F).^2,2),1), [], 1)) < 1e-5))
    end
  end

  c = 3;
  d = 3;
  X = randn(n, n, c, d);
  F = fft2(X);
  V = Fft2ToVec(F, mapping);

  % Test Fft2ToVecBackprop().

  lossfun = @(x)(0.5 * sum(x(:).^2));
  loss = lossfun(V);
  d_loss_V = V;
  d_loss_F = Fft2ToVecBackprop(d_loss_V);

  ep = 1e-5;
  n_loss_F = complex(nan(size(d_loss_F)), nan(size(d_loss_F)));
  for i = 1:ceil(numel(F) / 1000):numel(F)
    if ~(ismember(i, mapping.to_vec.real) && ismember(i, mapping.to_vec.imag))
      continue;
    end
    F_ = F;
    F_(i) = F_(i) + complex(ep, 0);
    V_ = Fft2ToVec(F_, mapping);
    loss_ = lossfun(V_);
    nF_real = (loss_ - loss)/ep;

    F_ = F;
    F_(i) = F_(i) + complex(0, ep);
    V_ = Fft2ToVec(F_, mapping);
    loss_ = lossfun(V_);
    nF_imag = (loss_ - loss)/ep;
    n_loss_F(i) = complex(nF_real, nF_imag);
  end

  d_loss_F = d_loss_F(~isnan(n_loss_F));
  n_loss_F = n_loss_F(~isnan(n_loss_F));

  errs = [abs(real(d_loss_F(:)) - real(n_loss_F(:))); ...
          abs(imag(d_loss_F(:)) - imag(n_loss_F(:)))];
  assert((max(errs) / (n^2)) < 1e-5)

  % Test VecToFft2Backprop().

  lossfun = @(x)(0.5 * (sum(real(x(:)).^2) + sum(imag(x(:)).^2)));
  loss = lossfun(F);
  d_loss_F = F;
  d_loss_V = VecToFft2Backprop(d_loss_F);

  ep = 1e-5;
  n_loss_V = nan(size(d_loss_V));
  for i = 1:ceil(numel(V) / 1000):numel(V)
    V_ = V;
    V_(i) = V_(i) + ep;
    F_ = VecToFft2(V_, mapping);
    loss_ = lossfun(F_);
    n_loss_V(i) = (loss_ - loss)/ep;
  end

  d_loss_V = d_loss_V(~isnan(n_loss_F));
  n_loss_V = n_loss_V(~isnan(n_loss_F));

  errs = abs(d_loss_V(:) - n_loss_V(:));
  assert((max(errs) / (n^2)) < 1e-5)

end

fprintf('Tests Passed\n');
