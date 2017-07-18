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
