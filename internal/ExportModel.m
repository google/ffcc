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

function ExportModel(model_file_path)
% ExportModel -- Export a trained model to a C++ project
% Arguments:
%
%  model_filename file path to a trained model for exporting

% Export the models to the binary files
load(model_file_path);

fft_size = size(model.F_fft,1);
% TODO(barron/yuntatsai): Remove gain maps, as they are now disabled.
gain = ones(size(model.B));
bias = model.B;
% TODO(barron/yuntatsai): Remove confidence, as it is now disabled.
confidence = zeros(size(model.B));

% Export Black Body & FFT models
% Write out binary format. These files are corresponding to the files
% under: //googlex/gcam/awb/core/data/
% awb_table_bias_*.bin
fp = fopen('/tmp/gain.bin', 'wb');
for y = 1:fft_size
    for x = 1:fft_size
        fwrite(fp, single(gain(y, x)), 'float');
    end
end
fclose(fp);

fp = fopen('/tmp/bias.bin', 'wb');
for y = 1:fft_size
    for x = 1:fft_size
        fwrite(fp, single(bias(y, x)), 'float');
    end
end
fclose(fp);

if ~isempty(confidence)
  % awb_table_bias_log_odds_ratio_*.bin
  fp = fopen('/tmp/confidence.bin', 'wb');
  for y = 1:fft_size
    for x = 1:fft_size
      fwrite(fp, single(confidence(y, x)), 'float');
    end
  end
  fclose(fp);
end

% awb_table_models_dft_*.bin
fp = fopen('/tmp/model_dft.bin', 'wb');
F_fft_repacked = RepackFourierModel(model.F_fft);

for c = 1:size(F_fft_repacked, 3)
  for y = 1:size(F_fft_repacked, 1)
    for x = 1:size(F_fft_repacked, 2)
      fwrite(fp, single(F_fft_repacked(y, x, c)), 'float');
    end
  end
end

fclose(fp);
