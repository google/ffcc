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

function data = PrecomputeTrainingData(folder, params)
% Reading training data from a folder (hardware statistics + illuminant) or
% cell array of folders, and convert it to the proper format.

% If 'folder' is actually a cell array of folders, then we recursively call
% PrecomputeTrainingData() on each subfolder and concatenate the output of
% each call.
if iscell(folder)

  data = {};
  for i_folder = 1:length(folder)
    data{i_folder} = PrecomputeTrainingData(folder{i_folder}, params);
  end
  data = cat(1, data{:});

else

  rng(0)

  print_name = folder(1+find(folder(1:end-1) == '/', 1, 'last'):end);

  assert(exist(folder, 'dir') > 0);

  % Grab all png images in this folder and in any sub-folder.
  image_filenames = ScrapeFolder(folder, '*.png');

  % Omit image filenames in an "augmented" subfolder.
  image_filenames = image_filenames( ...
    cellfun(@(x) isempty(strfind(x, '/face/')), image_filenames));

  % Subsample the training data, which usually should only happen during
  % debugging.
  if params.DEBUG.DATA_SUBSAMPLE > 1
    fprintf('WARNING: Subsampling the training data by a factor of %d\n', ...
      params.DEBUG.DATA_SUBSAMPLE);
    image_filenames = image_filenames( ...
      1:params.DEBUG.DATA_SUBSAMPLE:length(image_filenames));
  end

  if params.TRAINING.CROSSVALIDATION.NUM_FOLDS == 1
    % If we only have one cross-validation fold, then all images are in the
    % training set.
    is_test = sparse(length(image_filenames), 1, false);
  else
    % For some datasets the user may want to pre-specify which images lie in
    % which cross-validation fold. If a txt file exists with that specification
    % then we use it, otherwise we arbitrary separate the data into folds
    % according to the parameters.
    cvfolds_filename = fullfile(folder, 'cvfolds.txt');
    if exist(cvfolds_filename, 'file')
      fprintf('Using crossvalidation file %s\n', cvfolds_filename);
      % Cross-validation folds are described as a vector of integers, where
      % the fold in which each datapoint should be put in the test set is
      % indicated by that integer (one-indexed). See scripts/GehlerShiImport.m
      % to see how these values are generated.
      cvfoldidx = load(cvfolds_filename);
      cvfoldidx = cvfoldidx(1:params.DEBUG.DATA_SUBSAMPLE:length(cvfoldidx));
      assert(params.TRAINING.CROSSVALIDATION.NUM_FOLDS == max(cvfoldidx));
      assert(length(image_filenames) == length(cvfoldidx));
      is_test = sparse(1:length(cvfoldidx), cvfoldidx, true);
    else
      % Separate the data into NUM_FOLDS folds for cross-validation, where
      % consecutive images are assigned to folds in chunks of size CHUNK_SIZE.
      i = 1:length(image_filenames);
      j = mod(floor((i-1) ./ min(params.TRAINING.CROSSVALIDATION.CHUNK_SIZE, ...
            ceil((length(image_filenames)-1) / ...
              params.TRAINING.CROSSVALIDATION.NUM_FOLDS))), ...
            params.TRAINING.CROSSVALIDATION.NUM_FOLDS) + 1;
      is_test = sparse(i, j, true);
    end
  end

  data = {};
  fprintf('Loading %s (%d)', print_name, length(image_filenames));
  for i_file = 1:length(image_filenames)

    PrintDots(i_file, length(image_filenames), max(3, 71 - length(print_name)))
    image_filename = image_filenames{i_file};

    if params.SENSOR.LINEAR_STATS
      % Read the hardware statistics or sensor-space image, which is
      % assumed to be linear.
      I_linear = imread(image_filename);

      % The user can choose to ignore pixels with zeros in any channel (useful
      % for training on images with masked color charts or saturated pixels).
      if params.HISTOGRAM.MASK_ZERO_PIXELS
        I_valid = all(I_linear > 0,3);
      else
        I_valid = true(size(I_linear, 1), size(I_linear, 2));
      end

      % Compute the feature histogram.
      start_time = clock;
      X = FeaturizeImage(I_linear, I_valid, params);
      feature_time = etime(clock, start_time);
    else
      % Read the SRGB image.
      I_srgb = imread(image_filename);

      % The user can choose to ignore pixels with zeros in any channel (useful
      % for training on images with masked color charts or saturated pixels).
      if params.HISTOGRAM.MASK_ZERO_PIXELS
        I_valid = all(I_srgb > 0,3);
      else
        I_valid = true(size(I_srgb,1), size(I_srgb,2));
      end

      % Compute the feature histogram.
      start_time = clock;
      [X, ~, I_linear] = FeaturizeImageSrgb(I_srgb, I_valid, params);
      feature_time = etime(clock, start_time);
    end

    % Read the illuminant. The illuminant image is assumed to be specified in a
    % .txt file with the same path as the image.
    illuminant_filename = [image_filename(1:end-3), 'txt'];
    L = load(illuminant_filename);
    Y = RgbToUv(L);

    bins = EnumerateBins(params);
    if ~params.DEBUG.GRAY_WORLD_UNWRAPPING ...
        && (any(Y < bins(1)) || any(Y > bins(end)))
      fprintf('WARNING: A ground truth white point appear to lie ');
      fprintf('outside of the histogram range.\n');
      fprintf('This may cause a catastrophic failure unless gray-world ');
      fprintf('unwrapping is used.\n')
    end

    ccm_filename = [image_filename(1:end-4), '_ccm.txt'];
    if exist(ccm_filename, 'file')
      CCM = load(ccm_filename);
    else
      CCM = nan(3,3);
    end

    feature_vec = [];
    if params.DEEP.ENABLED
      for i_feat = 1:length(params.DEEP.FEATURE_FILENAME_TAGS)
        feature_filename = [image_filename(1:end-4), '_', ...
          params.DEEP.FEATURE_FILENAME_TAGS{i_feat}, '.txt'];
        feature_vec = [feature_vec; load(feature_filename)];
      end
    end

    filename_filename = [image_filename(1:end-4), '_filename.txt'];
    if exist(filename_filename, 'file')
      fid = fopen(filename_filename, 'r');
      original_filename = fgetl(fid);
      fclose(fid);
    else
      original_filename = '';
    end

    avg_rgb = squeeze(mean(mean(I_linear,1),2));
    avg_rgb = avg_rgb / sqrt(sum(avg_rgb.^2));

    data{i_file} = struct( ...
      'X', X, ...
      'Y', Y, ...
      'CCM', CCM, ...
      'feature_vec', feature_vec, ...
      'is_test', is_test(i_file, :), ...
      'filename', image_filename, ...
      'original_filename', original_filename, ...
      'feature_time', feature_time, ...
      'avg_rgb', avg_rgb);

    if i_file > 1
      assert(length(data{i_file}.feature_vec) ...
          == length(data{i_file-1}.feature_vec));
    end

    if params.DEBUG.RENDER_DATA
      try
        % Render the input image and the true white-balanced image.
        figure(10);
        if ~isa(I_linear, 'double')
          I_linear = ...
            double(I_linear) / double(intmax(class(I_linear)));
        end
        I_linear_white = bsxfun(@times, double(I_linear), ...
          min(L) ./ permute(L(:), [2,3,1]));
        imagesc(ApplySrgbGamma([I_linear, I_linear_white]));
        axis image off; drawnow;
      catch me
        fprintf(me.message)
      end
    end
  end

  data = cat(1, data{:});

  cv_folds = cat(1, data.is_test);
  n_cv_folds = full(sum(cv_folds,1));
  n_cv_spread = max(n_cv_folds) - min(n_cv_folds);
  if n_cv_spread > 8
    fprintf('WARNING: The sizes of the cross validation folds ( ');
    fprintf('%d, ', n_cv_folds);
    fprintf(') have a wide spread.');
  end

  % Check that all ground-truth white points are contained within the
  % histogram coordinates.
  Ys = cat(2, data.Y);
  uv_range = [min(Ys(:)), max(Ys(:))];
  uv_span = uv_range(2) - uv_range(1);
  if uv_span > (params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE)
    fprintf('WARNING: The white points spanned in the training set is ');
    fprintf('too large to be modeled by the histogram specified in the ');
    fprintf('coordinates.\n');
  end

  % Check if the histogram could benefit from being shifted to a new
  % starting position.
  uv_center = mean(uv_range);
  uv_lo = uv_center-(params.HISTOGRAM.BIN_SIZE*params.HISTOGRAM.NUM_BINS/2);
  uv_lo = round(uv_lo/params.HISTOGRAM.BIN_SIZE)*params.HISTOGRAM.BIN_SIZE;
  if abs(uv_lo - params.HISTOGRAM.STARTING_UV) > 1e-3
    fprintf('WARNING: params.HISTOGRAM.STARTING_UV appears to be ');
    fprintf('suboptimal, consider changing it from %f to %f\n', ...
      params.HISTOGRAM.STARTING_UV, uv_lo);
  end

  if params.DEBUG.RENDER_DATA
    % Center each histogram according to the ground-truth white point.
    Xs_shifted = {};
    for i = 1:length(data)
      uv = data(i).Y;
      [~, idx1] = min(abs(EnumerateBins(params) - uv(1)));
      [~, idx2] = min(abs(EnumerateBins(params) - uv(2)));
      Xs_shifted{i} = circshift(data(i).X, ...
        [params.HISTOGRAM.NUM_BINS/2-idx1, params.HISTOGRAM.NUM_BINS/2-idx2]);
    end
    Xs_shifted = cat(4, Xs_shifted{:});

    % Concatenate and render the average histogram and 7 randomly-chose
    % histograms, all centered.
    X_avg = mean(Xs_shifted,4);
    r = randperm(size(Xs_shifted,4));
    Xs_rand = Xs_shifted(:,:,:,r(1:7));
    Xs_vis = cat(4, X_avg, Xs_rand);
    Xs_vis = bsxfun(@rdivide, double(Xs_vis), max(max(Xs_vis, [], 1), [], 2));
    try
      figure(11)
      imagesc(reshape(permute((Xs_vis), [1,3,2,4]), ...
        size(Xs_vis,1)*size(Xs_vis,3), [])); axis image off
      title('Average Centered Histogram | 7 Random Centered Histograms')
      drawnow;
    catch me;
      me;
    end

    % Render the ground-truth white points in (u, v) coordinates relative to
    % a "do nothing" [2,1,2] white point which just halves green.
    [us, vs] = ndgrid(EnumerateBins(params));
    rgb = cat(3, -us, -log(2) * ones(size(us)), -vs); % gain down green by 2
    rgb = exp(bsxfun(@minus, rgb, max(rgb, [], 3)));
    rgb = bsxfun(@rdivide, rgb, max(rgb,[],3));
    try
      figure(12);
      imagesc(EnumerateBins(params), EnumerateBins(params), rgb);
      axis image; hold on; scatter(Ys(2,:), Ys(1,:))
    catch me;
      me;
    end
  end
end
