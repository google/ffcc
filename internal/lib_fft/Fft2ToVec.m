function V = Fft2ToVec(F, mapping, apply_scaling)
% Takes a K*K*N*M complex-valued tensor of images/filters F in Fourier-domain
% and returns a K^2*N*M real-valued tensor V, where each square image/vector in
% F corresponds to a vector in F.
% Allows for optional input arguments which specify a faster lookup table,
% precomputed by Fft2ToVecPrecompute().

if nargin < 2
  mapping = [];
end

if ~exist('apply_scaling', 'var')
  apply_scaling = true;
end

% Currently only square images are supported.
assert(size(F,1) == size(F,2));

% Only power-of-two sized images are supported.
assert(log2(size(F,1)) == round(log2(size(F,1))))

% Currently only two outer dimensions of V are supported.
assert(size(F,1) * size(F,2) * size(F,3) * size(F,4) == numel(F));

n = size(F,1);

if numel(F) > (size(F,1) * size(F,2))
  V = zeros(size(F,1) * size(F,2), size(F,3), size(F,4));
  for d = 1:size(F,4)
    for c = 1:size(F,3)
      V(:,c,d) = Fft2ToVec(F(:,:,c,d), mapping, apply_scaling);
    end
  end
else
  if ~isempty(mapping) % If a lookup table is given as input, use that.
    if apply_scaling
      F([1, n/2+1], [1, n/2+1]) = ...
        complex(real(F([1, n/2+1], [1, n/2+1])) / sqrt(2), ... % scale reals,
        imag(F([1, n/2+1], [1, n/2+1]))); % but pass the imaginary component.
    end
    V = [real(F(mapping.to_vec.real)); imag(F(mapping.to_vec.imag))];
    if apply_scaling
      V = V * sqrt(2);
    end
  else % Otherwise vectorize the FFT manually.
    F_real = real(F);
    F_imag = imag(F);
    if apply_scaling
      F_real([1, n/2+1], [1, n/2+1]) = F_real([1, n/2+1], [1, n/2+1]) / sqrt(2);
    end

    V = cat(1, ...
      F_real(1:(n/2+1), 1), ...
      F_real(1:(n/2+1), n/2+1), ...
      reshape(F_real(:, 2:(n/2)), [], 1), ...
      F_imag(2:(n/2),1), ...
      F_imag(2:n/2, n/2+1), ...
      reshape(F_imag(:,2:(n/2)), [], 1));

    if apply_scaling
      V = V * sqrt(2);
    end
  end
end
