% This is a stand-alone script for calibrating the Cheng color constancy
% dataset, by estimating the CCM of each camera in the dataset as well as the
% illuminant color for each image. This should probably never be run by a user
% of this codebase, please contact barron@ before running this script.

clearvars;

addpath('../internal/')

paths = DataPaths;
cheng_folder = paths.cheng;

output_folder = '../projects/Cheng/tags/';
mkdir(output_folder);

% The names of the cameras in the Cheng dataset. This could be scraped from the
% folder names in cheng_folder, but we are omitting one camera with too few
% images (as is suggested by the dataset).
CAMERA_NAMES = { ...
  'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200', ...
  'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'};

% The RGB values of the Macbeth color chart. The CCMs and gains are computed
% to minimize error with respect to these values, so how these values are
% selected will directly affect how the CCMs look.
macbeth_constants = MacbethConstants;

data = {};
for i_camera = 1:length(CAMERA_NAMES)
  camera_name = CAMERA_NAMES{i_camera};

  % Load the ground-truth illuminants (we use the ground-truth produced by
  % the creators of the dataset, not our own re-estimated ground-truth) as well
  % as the black level and saturation level for this camera.
  gt = load(fullfile(cheng_folder, camera_name, [camera_name, '_gt.mat']));
  black_level = gt.darkness_level;
  saturation = gt.saturation_level;

  images_folder = fullfile(cheng_folder, camera_name, 'PNG');
  masks_folder = fullfile(cheng_folder, camera_name, 'CHECKER');

  % loop through each image.
  dirents = dir(fullfile(images_folder, '*.PNG'));
  for i_file  = 1:length(dirents)
    PrintDots(i_file, length(dirents), 80)

    % The Cheng dataset convenient provides pre-processed the RGB values in the
    % color checker, which we read in.
    root_filename = dirents(i_file).name;
    root_filename = root_filename(1:(find(root_filename == '.', 1, 'first')-1));
    rgb = load(fullfile(cheng_folder, camera_name, 'CHECKER', ...
      [root_filename, '_color.txt']));
    rgb = max(0, (rgb - black_level)) ./ (saturation - black_level);
    rgb_flip = flipud(permute(reshape(permute(rgb, [1,3,2]), 6,4,3), [2,1,3]));
    rgb = reshape(ipermute(rgb_flip, [2,1,3]), [], 3);

    % Values > 0.98 are omitted by setting them to NaN.
    rgb(repmat(any(rgb >= 0.98,2), 1,3)) = nan;

    try
      image_chart = ApplySrgbGamma(permute( ...
        reshape(permute(max(0, min(1, rgb)), [1,3,2]), 6,4,3), [2,1,3]));
      true_chart = permute( ...
        reshape(permute(macbeth_constants.SRGB, [1,3,2]), 6,4,3), [2,1,3]);
      figure(2);
      imagesc([image_chart; true_chart]); axis image off;
      drawnow;
    catch me
      me;
    end
    data{end+1} = struct( ...
      'rgb', rgb, ...
      'camera_name', camera_name, ...
      'filename', dirents(i_file).name, ...
      'gt_illum', gt.groundtruth_illuminants(i_file,:));
  end
end
data = cat(1, data{:});


image_CCMs = cell(length(data), 1);
image_gains = cell(length(data), 1);
sensor_CCMs = struct();
camera_names = unique({data.camera_name});
for i_camera = 1:length(camera_names)
  camera_name = camera_names{i_camera};
  camera_valid = cellfun(@(x) strcmp(x, camera_name), {data.camera_name});
  n_data = nnz(camera_valid);
  rgbs = {data(camera_valid).rgb};

  % Initialize the CCM for the camera. It is convenient to work on the
  % transposed CCM, indicated with _t.
  CCM_t = eye(3);

  % Iteratively re-estimate each image's gain using the CCM, and then
  % re-estimate the CCM using the gained images.
  for iter = 1:20
    gains = {};
    for i_image = 1:length(rgbs)
      gain = zeros(1,3);
      rgb = rgbs{i_image};
      % What the color chart should look like in sensor space, given our
      % current estimate of the CCM.
      rgb_true = macbeth_constants.RGB * inv(CCM_t);
      for c = 1:3
        x = rgb(:,c);
        y = rgb_true(:,c);
        v = ~isnan(x) & (x > 0);
        % We want a least squares solve of x(v) \ y(v), which is equivalent to:
        gain(c) = sum(x(v) .* y(v)) ./ sum(x(v) .* x(v));
        assert(gain(c) > 0);
      end
      gains{i_image} = gain;
    end

    % Tint each image according to the estimated gains.
    rgbs_gained = {};
    for i_image = 1:length(rgbs)
      rgbs_gained{i_image} = bsxfun(@times, gains{i_image}, rgbs{i_image});
    end

    % Construct the linear system of equations to solve for a (transposed) CCM.
    A = cat(1, rgbs_gained{:});
    B = repmat(macbeth_constants.RGB, n_data, 1);
    valid = ~any(isnan(A), 2);

    % Approximately constrain the CCM to be row-normalized by appending
    % a strong unit-norm penalty.
    lambda = 1e8;
    Ac = [A(valid,:); lambda * ones(1,3)];
    Bc = [B(valid,:); lambda * ones(1,3)];

    % Solve the system.
    CCM_t = Ac \ Bc;

    % Row-normalize the CCM just in case the penalty wasn't strong enough.
    CCM_t = bsxfun(@rdivide, CCM_t, sum(CCM_t,1));

    % Compute the loss (which was theoretically the loss being minimized when
    % estimating CCM_t.
    loss = sum(sum((A(valid,:) * CCM_t - B(valid,:)).^2));
    fprintf('%02d: %f\n', iter, loss);

    try
      % Visualize our current estimate of the balanced color charts against
      % the ground-truth color chart.
      figure(2)
      Y = A * CCM_t;
      Y(isnan(Y)) = 0;
      imagesc(ApplySrgbGamma(max(0, min(1, reshape([Y;B], 24, [], 3)))));
      axis off;
      drawnow;
    catch me
      me;
    end
  end

  CCM = CCM_t';
  image_gains(camera_valid) = gains;
  image_CCMs(camera_valid) = {CCM};
  sensor_CCMs.(camera_name) = CCM;
end
image_gains = cat(1, image_gains{:});

% Sanity-check that our estimated gains are roughly consistent with the
% ground-truth illuminants provided by the dataset.
old_illuminants = bsxfun(@rdivide, cat(1, data.gt_illum), ...
  sqrt(sum(cat(1, data.gt_illum).^2,2)));
new_illuminants = bsxfun(@rdivide, ...
  1./image_gains, sqrt(sum((1./image_gains).^2,2)));
errors = acos(sum(new_illuminants .* old_illuminants,2)) * 180/pi;
assert(median(errors) < 1.5)
assert(max(errors) < 8)

% Save the gains and the CCMs to disk.
save(fullfile(output_folder, 'illuminants.txt'), 'new_illuminants', ...
  '-ascii', '-double')

Cs = {};
for i_camera = 1:length(camera_names)
  camera_name = camera_names{i_camera};
  CCM = sensor_CCMs.(camera_name);
  % Note that these CCMs do not correspond to any particular "laboratory"
  % illuminantion (D65, D50, etc). They are whatever CCM best minimizes error
  % over the entire dataset, and so they are likely a compromise between many
  % "kinds" of CCMs.
  save(fullfile(output_folder, [camera_name, '_CCM.txt']), 'CCM', ...
    '-ascii', '-double')
  Cs{end+1} = CCM;
end

avg_CCM = mean(cat(3, Cs{:}),3);
avg_CCM = bsxfun(@rdivide, avg_CCM, sum(avg_CCM,2));

save(fullfile(output_folder, 'average_CCM.txt'), 'avg_CCM', ...
    '-ascii', '-double')
