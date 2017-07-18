% Copyright 2017 Google Inc.
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
%     https://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function UpdateGolden( project_name )
% Update golden will print out the values that should be updated for
% //googlex/gcam/awb/core:awb_test

pngs_folder = '../../core/tests/pngs';

addpath(genpath('./internal/'));
params = LoadProjectParams(project_name);

params.output_model_filename
load(params.output_model_filename);

F = model.F_fft;
B = model.B;
C = model.C;

pngs = dir(fullfile(pngs_folder, '*.png'));

fprintf('Copy the following and replace one of the entry under ');
fprintf('awb_test.cc: const std::array<GoldenDeviceEntry, 4> kDevices\n');
fprintf('------begin of the entry------\n');

fprintf('{k%s, {{\n', project_name);
for i = 1:numel(pngs)
  fn = pngs(i).name;
  im = uint8(imread(fullfile(pngs_folder, fn)));

  X = FeaturizeImage(im, [], params);

  [obs, confidence, ~, ~, ~] = EvaluateModel(F, B, C, X, fft2(X), ...
    [], [], [], params);
  fprintf('\t{{{%f, %f}}, %f}', obs.mu(1), obs.mu(2), confidence);
  if i < numel(pngs)
    fprintf(',\n');
  else
    fprintf('}}}\n');
  end
end

fprintf('-------end of the entry--------\n');

end

