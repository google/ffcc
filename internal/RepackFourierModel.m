function [ output ] = RepackFourierModel( F_fft )
% Repack the Fourier coefficients to be compatible with the Halide
% implementation format

half_fft_size = size(F_fft, 1) / 2 + 1;

output = {};
for c = 1:size(F_fft, 3)
  output{end + 1} = single(real(F_fft(1:half_fft_size, :, c)));
  output{end + 1} = single(imag(F_fft(1:half_fft_size, :, c)));
end

  output = cat(3, output{:});
end

