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

function params = PrivateConstants
% Default settings for the AWB algorithm which are unlikely to change for
% different projects.

% The location of the learned models, used by Train().
params.TRAINING.OUTPUT_MODEL_FOLDER = 'models/';
warning off all
system(['mkdir ', params.TRAINING.OUTPUT_MODEL_FOLDER, ' &> /dev/null']);
warning on all;

% The location of visualizations of each model, used by Visualize().
params.TRAINING.OUTPUT_VISUALIZATION_ROOT = 'vis/';
warning off all
system(['mkdir ', params.TRAINING.OUTPUT_VISUALIZATION_ROOT, ' &> /dev/null']);
warning on all;

% Images in these paths are used for training during cross-validation and in
% final training, but never in the test-set of any cross-validation fold.
params.TRAINING.EXTRA_TRAINING_DATA_FOLDERS = { ...
};

% Images in these paths are used in every test-set of any cross-validation fold
% and during visualization, but never for training.
params.TRAINING.EXTRA_TESTING_DATA_FOLDERS = { ...
};

% If true, causes visualization to dump extensive output, which is useful for
% formatting figures for papers, etc.
params.TRAINING.DUMP_EXHAUSTIVE_VISUALIZATION = false;

% If true, pre-compute training data from the Gehler Shi dataset for use in
% training this project. For some projects this data is not needed, and so this
% can be set to false to save disk space and time during precomputation.
params.TRAINING.GENERATE_GEHLERSHI_DATA = true;

% The number of folds to split the cross-validation data into, 3 is a
% standard choice, but anywhere in [2, 10] is fine. Setting this to 1
% will cause cross-validation to train and test on all of the data, which
% is completely invalid in terms of actual performance, but is allowed
% because it gives an upper bound on performance. The number of folds
% determines the amount of training data used in each fold of cross-validation:
% (NUM_FOLDS-1)/NUM_FOLDS of the data will be used for training, and
% 1/NUM_FOLDS will be used for testing.
params.TRAINING.CROSSVALIDATION.NUM_FOLDS = 3;

% The number of adjacent images in the cross-validation data which will be
% assined to the same fold. If the cross-validation data is taken from a
% video sequence, or if adjacent images resemble each other, then this
% number should be large. If each image in the cross-validation data is
% statistically independent from its neighbors, then this value can be 1.
params.TRAINING.CROSSVALIDATION.CHUNK_SIZE = 3;

% A small value that controls the scaling of the confidence map produced
% during training. The output confidence is shifted and scaled such that
% output confidence must lie in the range of
% [CONFIDENCE_EPSILON, 1-CONFIDENCE_EPSILON].
params.TRAINING.CONFIDENCE_EPSILON = 0.001;

% The parameters control how many passes of LBFGS we perform as the loss
% function is annealed from convex option to non-convex. The number of
% iterations for each pass of LBFGS is controlled by the "INITIAL" and
% "FINAL" constants, such that the number of iterations for a pass is
% produced by linearly interpolating between those two values in log-space.
% Here we are using a subset of this functionality where we perform
% two passes of optimization, one to pretrain with the convex loss and the
% second to fine-tune with the non-convex loss.
params.TRAINING.NUM_ITERS_ANNEAL = 2;
params.TRAINING.NUM_ITERS_LBFGS_INITIAL = 16;
params.TRAINING.NUM_ITERS_LBFGS_FINAL = 64;

% If true, use bilinear interpolation to construct the ground-truth PDF used in
% the pretraining loss.
params.TRAINING.SMOOTH_CROSS_ENTROPY = true;

% A value that is added to the diagonal of the covariance matrix in the fitted
% Von Mises distributions when evaluating and training our model, in units of
% histogram bins. Setting this to 1 seems to work well, though analytically this
% should be 1/12, which is the variance of the histogram being sampled.
params.HISTOGRAM.VON_MISES_DIAGONAL_MODE = 'pad';

% These parameters determine how each parameter will be varied during
% coordinate descent. For each parameter x being searched we initialize its
% tuning multiplier to INIT_MULT. If we can improve cross-validation error
% by modifying that parameter, we scale up its multiplier by raising it to
% MULT_INCREASE (assumed to be >= 1), and if not we scale it down by raising it
% to MULT_DECREASE (assumed to be =< 1). Coordinate descent terminates after
% NUM_ITERS passes over all variables, or when all multipliers are <= MIN_MULT.
params.TUNING.MULT_INCREASE = 1;
params.TUNING.MULT_DECREASE = (1/2);
params.TUNING.INIT_MULT = 2;
params.TUNING.MIN_MULT = 2^(1/4);
params.TUNING.NUM_ITERS = 10;
params.TUNING.DISPLAY = true;
params.TUNING.ONLY_TUNE_FIELDS = {};

% Whether or not to print the LBFGS output during optimization.
params.DEBUG.DISPLAY_LBFGS = true;

% Whether or not to render visualizations of the training data during loading.
params.DEBUG.RENDER_DATA = false;

% Whether or not to render the current model during training.
params.DEBUG.RENDER_MODEL = false;

% If RENDER_MODEL==true, the number of seconds to wait between renderings.
params.DEBUG.RENDER_PERIOD = 30;

% How much to subsample the data. Setting to 10 samples 1/10th of the data,
% setting to 1 samples all data, etc. Should be set to 1 for all real results,
% but can be set to larger values for running fast experiments.
params.DEBUG.DATA_SUBSAMPLE = 1;

% If true, use finite-differencing to check that the analytic
% gradient returned by the loss function is close to the numerical
% gradient.
params.DEBUG.CORRECT_GRADIENT = false;

% If true, confirm that the preconditioner is an accurate Jacobi preconditioner
% of the regularizer, by printing out a series of errors which should be close
% close to zero.
params.DEBUG.CORRECT_PRECONDITIONER = false;

% If true, randomly sample the model weight space to check that the
% loss is convex.
params.DEBUG.CONVEX_LOSS = false;

% If true, randomly sample models and data partitions to check that the
% loss is additive, that is, that:
%    loss( {data_a} U {data_b} ) = loss( {data_a} ) + loss( {data_b} )
params.DEBUG.ADDITIVE_LOSS = false;

% If true, use a gray-world prior to "unwrap" the white point estimate. If
% false use the default gray-light prior.
params.DEBUG.GRAY_WORLD_UNWRAPPING = false;

% The minimum linear image intensity (in [0,1]) for which we're willing to trust
% input pixels when constructing histograms.
params.HISTOGRAM.MINIMUM_INTENSITY = 1/256;

% The loss imposed on the von Mises non-pretraining stage.
% Valid options: 'likelihood', 'squared_error', 'expected_squared_error'
params.TRAINING.VON_MISES_LOSS = 'likelihood';

% If true, constrains the von mises distributions fitted during training and
% testing to be isotropic (ie, their convariance matrices are scaled identity
% matrices).
params.TRAINING.FORCE_ISOTROPIC_VON_MISES = false;

% If true, enable the (experimental) deep learning mode, which allows external
% features to be used during training.
params.DEEP.ENABLED = false;

% When the deep mode is enabled, this tag is used to select which feature
% filename to use as input to the model.
params.DEEP.FEATURE_FILENAME_TAG = 'feature';

% If true, do not compute the regularization cost. Should be turned on iff
% The "deep" mode is enabled, because weight decay plays a similar (but better)
% role to the regularizer.
params.TRAINING.DISABLE_REGULARIZER = false;

% If true, flag runtimes as invalid (used when a model depends on features whose
% runtimes are not known).
params.TRAINING.TIMES_ARE_INVALID = false;

% If non-empty, a plain text filename containing per-image cross-validation
% RGB output values.
params.DEBUG.OUTPUT_SUMMARY = [];
