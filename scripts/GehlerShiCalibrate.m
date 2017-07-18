% This is a stand-alone script for calibrating the Gehler-Shi color constancy
% dataset, by estimating the CCM of each camera in the dataset as well as the
% illuminant color for each image. This should probably never be run by a user
% of this codebase, please contact barron@ before running this script.

clearvars;

addpath('../internal/')


paths = DataPaths;

% Set the necessary paths to parse the training data.
root_folder = paths.gehler_shi;
images_folder = fullfile(root_folder, 'images/');
coordinates_folder = fullfile(root_folder, 'groundtruth/coordinates/');
illuminant_filename = fullfile(root_folder, 'groundtruth/real_illum_568.mat');
cvfold_filename = fullfile(root_folder, 'threefoldCVsplit.mat');
output_folder = '../projects/GehlerShi/tags/';

% The properties of the sensors used in the Gehler-Shi dataset, which are
% necessary to correctly read and normalize the images.
gehlershi_sensor = GehlerShiSensor;

% The RGB values of the Macbeth color chart. The CCMs and gains are computed
% to minimize error with respect to these values, so how these values are
% selected will directly affect how the CCMs look.
macbeth_constants = MacbethConstants;

dirents = dir(fullfile(images_folder, '*.png'));
data = {};
for i_file  = 1:length(dirents)
  PrintDots(i_file, length(dirents))

  root = dirents(i_file).name;
  root = root(1:(find(root == '.', 1, 'first')-1));

  im_filename = fullfile(images_folder, dirents(i_file).name);

  [image, ~, camera_name] = ReadImage(im_filename);

  image_vec = reshape(image, [], 3);

  assert(all(image(:) >= 0));

  % Load and rescale the color checker coordinates.
  cc_coord = load(fullfile(coordinates_folder, [root '_macbeth.txt']));
  scale = cc_coord(1,[2 1])./[size(image,1) size(image,2)];
  cc_coord = [cc_coord(6:end,1)/scale(1), cc_coord(6:end,2)/scale(2)];

  % For each color in the checker, estimate the average RGB value.
  rgb = {};
  for i = 1:24
    mask = roipoly(image, cc_coord((i-1) * 4 + [1:4],1), ...
                          cc_coord((i-1) * 4 + [1:4],2));
    mask_vec = mask(:);
    avg_val = mean(image_vec(mask_vec,:));

    % If any pixel in the check is nearly saturated, ignore it's average color
    % by setting it to NaN.
    max_val = max(image_vec(mask_vec,:), [], 1);
    if any(max_val >= 0.98)
      avg_val = nan(1,3);
    end

    rgb{i} = avg_val;
  end
  rgb = cat(1, rgb{:});

  try
    image_chart = ApplySrgbGamma( ...
      permute(reshape(permute(max(0, min(1, rgb)), [1,3,2]), 6,4,3), [2,1,3]));
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
    'filename', dirents(i_file).name);
end
data = cat(1, data{:});

fixed_image_CCMs = cell(length(data), 1);
fixed_image_gains = cell(length(data), 1);
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
        v = ~isnan(x);
        % We want a least squares solve of x(v) \ y(v), which is equivalent to:
        gain(c) = sum(x(v) .* y(v)) ./ sum(x(v) .* x(v));
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
  fixed_image_gains(camera_valid) = gains;
  fixed_image_CCMs(camera_valid) = {CCM};
  sensor_CCMs.(camera_name) = CCM;
end
fixed_image_gains = cat(1, fixed_image_gains{:});

% Sanity-check that our estimated gains are roughly consistent with the
% ground-truth illuminants provided by the dataset.
gt = load(illuminant_filename);
old_illuminants = bsxfun(@rdivide, gt.real_rgb, sqrt(sum(gt.real_rgb.^2,2)));
new_illuminants = bsxfun(@rdivide, ...
  1./fixed_image_gains, sqrt(sum((1./fixed_image_gains).^2,2)));
errors = acos(sum(new_illuminants .* old_illuminants,2)) * 180/pi;

 % TODO(barron): investigate why the maxium error is pretty large.
assert(median(errors) < 1)
assert(max(errors) < 10)

% Save the gains and the CCMs to disk.
filename = fullfile(output_folder, 'illuminants.txt');
system(['g4 edit ', filename]);
save(filename, 'new_illuminants', ...
  '-ascii', '-double')

for i_camera = 1:length(camera_names)
  camera_name = camera_names{i_camera};
  CCM = sensor_CCMs.(camera_name);
  % Note that these CCMs do not correspond to any particular "laboratory"
  % illuminantion (D65, D50, etc). They are whatever CCM best minimizes error
  % over the entire dataset, and so they are likely a compromise between many
  % "kinds" of CCMs.
  filename = fullfile(output_folder, [camera_name, '_CCM.txt']);
  system(['g4 edit ', filename]);
  save(filename, 'CCM', '-ascii', '-double')
end

% Compute two average CCMs from the dataset, one that is a naive average of the
% two, and the other weighted by the number of images.
CCM_avg = (sensor_CCMs.Canon1D + sensor_CCMs.Canon5D)/2;
CCM_avg = bsxfun(@rdivide, CCM_avg, sum(CCM_avg,2));

weight_1D = mean(cellfun(@(x)(strcmp(x, 'Canon1D')), {data.camera_name}));
CCM_weighted = ...
  weight_1D * sensor_CCMs.Canon1D + (1-weight_1D) * sensor_CCMs.Canon5D;
CCM_weighted = bsxfun(@rdivide, CCM_weighted, sum(CCM_weighted,2));

filename = fullfile(output_folder, 'average_CCM.txt');
system(['g4 edit ', filename]);
save(filename, 'CCM_avg', '-ascii', '-double')

filename = fullfile(output_folder, 'weighted_CCM.txt');
system(['g4 edit ', filename]);
save(filename, 'CCM_weighted', '-ascii', '-double')
