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

function data_augmented = PrecomputeMixedData( ...
  crossvalidate_folder, extra_training_folders, extra_testing_folders, params)
% Loads a primary dataset used for cross-validation, while also loading
% some number of "extra" datasets, which are flagged as "extra" and are
% used in the training set of every cross-validation fold. This allows us
% to experiment with domain adaptation across sensors and datasets.

if isempty(params.TRAINING.CROSSVALIDATION_DATA_FOLDER)
  assert(params.TRAINING.CROSSVALIDATION.NUM_FOLDS == 0)
  data_cv = [];
  fold_sz = [1, 1];
else
  data_cv = PrecomputeTrainingData(crossvalidate_folder, params);
  fold_sz = size(data_cv(1).is_test);
end

if ~isempty(extra_training_folders)
  data_extra_train = PrecomputeTrainingData(extra_training_folders, params);
  for i = 1:length(data_extra_train)
    data_extra_train(i).is_test = sparse(false(fold_sz));
  end
else
  data_extra_train = [];
end

if ~isempty(extra_testing_folders)
  data_extra_test = PrecomputeTrainingData(extra_testing_folders, params);
  for i = 1:length(data_extra_test)
    data_extra_test(i).is_test = sparse(true(fold_sz));
  end
else
  data_extra_test = [];
end
data_augmented = cat(1, data_cv, data_extra_train, data_extra_test);
