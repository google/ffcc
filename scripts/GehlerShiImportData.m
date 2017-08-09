% This is a stand-alone script for importing the Gehler-Shi color constancy
% dataset into the format specified by each project.

clearvars;
addpath(genpath('../internal/'));
addpath(genpath('../projects/'));
rng(0)

paths = DataPaths;

% Set the necessary paths to parse the training data.
gehlershi_folder = paths.gehler_shi;
images_folder = fullfile(gehlershi_folder, 'images/');
coordinates_folder = fullfile(gehlershi_folder, 'coordinates/');
illuminants_filename = fullfile(gehlershi_folder, 'real_illum_568.mat');
crossvalidationfolds_filename = ...
    fullfile(gehlershi_folder, './threefoldCVsplit.mat');

VISUALIZE = false;

% The multiplier on the saturation value for which we mask out bright
% pixels. Setting this to 1 actually uses the saturation value, but we
% would like to be a little conservative so we set it a bit lower.
SATURATION_SCALE = 0.95;

% Grab the names of all the projects.
project_names = dir('../projects/GehlerShi*');
project_names = {project_names.name};
project_names = project_names(cellfun(@(x) x(1) ~= '.', project_names));
project_names = project_names( ...
  cellfun(@(x) isempty(strfind(x, '.m')), project_names));

% Load all project parameters.
projects = [];
warning off all;
for i_project = 1:length(project_names)
  project_name = project_names{i_project};
  projects.(project_name) = LoadProjectParams(project_name);
  if ~projects.(project_name).TRAINING.GENERATE_GEHLERSHI_DATA
    projects = rmfield(projects, project_name);
  end
end
warning on all;

% Preallocate the data we'll collect.
data.filenames = {};
for project = fieldnames(projects)'
  project = project{1};
  data.images.(project) = {};
  data.illuminants.(project) = {};
  data.CCMs.(project) = {};
  data.camera_names.(project) = {};
  data.exifs.(project) = {};
end

% Load the ground-truth illuminants (we use the ground-truth produced by
% the creators of the dataset, not our own re-estimated ground-truth).
load(illuminants_filename);

% loop through each image.
dirents = dir(fullfile(images_folder, '*.png'));
for i_file  = 1:length(dirents)
  PrintDots(i_file, length(dirents), 80)
  im_filename = fullfile(images_folder, dirents(i_file).name);
  data.filenames{i_file} = im_filename;

  [image, mask, camera_name, illuminant, CCM_input, exif] ...
    = ReadImage(im_filename);
  image(repmat(~mask, [1,1,3])) = 0;

  root_filename = dirents(i_file).name;
  root_filename = root_filename(1:(find(root_filename == '.', 1, 'first')-1));

  if VISUALIZE
    try
      figure(1);
      % Make a white-balanced linear image with the ground-truth illuminant.
      gains = max(illuminant) ./ illuminant;
      image_vis = bsxfun(@times, image, permute(gains(:), [2,3,1]));
      image_vis = ...
        reshape(reshape(image_vis, [], 3) * CCM_input', size(image_vis));
      image_vis = image_vis ./ max(image_vis(:));
      image_vis = ApplySrgbGamma(max(0, min(1, image_vis)));
      imagesc(image_vis);
      title([num2str(i_file), ' - input'])
      axis image off; drawnow;
    catch me
      me;
    end
  end

  % Modify "image" and "mask" according to each project's specifications.
  for project = fieldnames(projects)'
    project = project{1};
    stats_size = projects.(project).SENSOR.STATS_SIZE;
    bit_width = projects.(project).SENSOR.STATS_BIT_DEPTH;

    image_project = image;
    mask_project = mask;
    illuminant_project = illuminant;

    if ~isnan(stats_size)
      % If the image and the stats have inverted aspect ratios, rotate the
      % image and mask.
      if (sign(log(size(image_project,1) / size(image_project,2))) ...
          ~= sign(log(stats_size(1) / stats_size(2))))
        image_project = cat(3, ...
          rot90(image_project(:,:,1)), ...
          rot90(image_project(:,:,2)), ...
          rot90(image_project(:,:,3)));
        mask_project = rot90(mask_project);
      end

      % Crop off the last row/column of the image so its size is divisible by 2,
      % if necessary.
      image_project = image_project(...
        1:floor(size(image_project,1)/2)*2, ...
        1:floor(size(image_project,2)/2)*2, :);

      mask_project = mask_project(...
        1:floor(size(mask_project,1)/2)*2, ...
        1:floor(size(mask_project,2)/2)*2, :);

      % Crop the image/mask to match the aspect ratio of the specified stats.
      im_size = [size(image_project,1), size(image_project,2)];
      scale = min(im_size ./ stats_size);
      crop_size = floor( (stats_size * scale)/2 ) * 2;
      image_project = image_project(...
        (im_size(1) - crop_size(1))/2 + [1:crop_size(1)], ...
        (im_size(2) - crop_size(2))/2 + [1:crop_size(2)], :);
      mask_project = mask_project(...
        (im_size(1) - crop_size(1))/2 + [1:crop_size(1)], ...
        (im_size(2) - crop_size(2))/2 + [1:crop_size(2)]);
      im_size = [size(image_project,1), size(image_project,2)];
    end

    if isnan(projects.(project).SENSOR.CCM)
      % If the project does not specify the CCM of the sensor, we inheret
      % the CCM of this image from the Gehler-Shi dataset.
      CCM_project = CCM_input;
      image_canonized = image_project;
      illuminant_canonized = illuminant_project;
    else
      % If the project specifies the CCM of the sensor, we need to construct a
      % "fake" illuminant and image from the input that looks as though it was
      % imaged by the project's CCM.
      CCM_project = projects.(project).SENSOR.CCM;

      % We can correct the illuminant by warping the illuminant or by warping
      % the gains. If the CCM is row-normalized (which it should be for any
      % camera, but isn't for the sRGB matrix which we support) then both
      % procedures are equivalent, but to support any CCM we transform the
      % illuminant two ways and average the two results.
      CCM_fix = CCM_project \ CCM_input;
      illuminant_canonized1 = CCM_fix * illuminant_project;
      illuminant_canonized2 = 1./ (CCM_fix * (1./illuminant_project));
      illuminant_canonized = UvToRgb(...
        (RgbToUv(illuminant_canonized1) + ...
         RgbToUv(illuminant_canonized2))/2);

      % Construct the 3x3 matrix correction to undo the true white point and
      % input CCM, and apply the project's CCM.
      T0 = CCM_input * diag(1./illuminant_project);
      T1 = CCM_project * diag(1./illuminant_canonized);
      T = T1 \ T0;

      % "Canonize" the input image using T.
      image_canonized = ...
        reshape(reshape(image_project, [], 3) * T', size(image_project));
    end

    if ~isnan(stats_size)
      % Check that the image and the stats have the same aspect ratio.
      assert(abs(log(im_size(2) / im_size(1)) ...
               - log(crop_size(2) / crop_size(1))) < 0.01)

      % Downsample the image according to the mask, and downsample the mask.
      % This downsample is weighted (or equivalently, done in homogenous
      % coordinates) so that masked pixels are ignored.
      downsample = @(x)imresize(x, stats_size, 'bilinear');

      image_numer = downsample(bsxfun(@times, image_canonized, mask_project));
      image_denom = downsample(double(mask_project));
      image_down = bsxfun(@rdivide, image_numer, max(eps, image_denom));

      % A small denominator means that very few masked pixels are in the
      % full-res image at that position, and so the low-res image should be
      % masked out at that position.
      DENOM_THRESHOLD = 0.01;
      mask_down = image_denom >= DENOM_THRESHOLD;
    else
      image_down = image_canonized;
      mask_down = mask_project;
    end

    % Zero out the masked values.
    image_down(repmat(~mask_down, [1,1,3])) = 0;

    clear mask_down;

    if strcmp(project, 'Photos')
      % For this project we would like to resynthesize a set of randomly tinted
      % images. PHOTOS_UV_SIGMA controls how saturated those images are, where
      % 0.15 appears to be empirically similar to the saturation of tints seen
      % in real-world data.
      PHOTOS_UV_SIGMA = 0.15;
      uv_tint = randn(2,1)*PHOTOS_UV_SIGMA;
      illuminant_tinted = UvToRgb(uv_tint);

      % Gain the image to make a white-balanced image and apply the
      % project's CCM.
      gains = max(illuminant_canonized)./illuminant_canonized;

      image_white = bsxfun(@times, image_down, permute(gains(:), [2,3,1]));
      image_white = max(0, ...
        reshape(reshape(image_white, [], 3) * CCM_project', size(image_white)));

      max_val = max(image_white(:));
      image_white = image_white ./ max_val;

      % Apply the random gains (in the CCM'ed space) to get a tinted image.
      gains = illuminant_tinted ./ min(illuminant_tinted);

      image_tinted = bsxfun(@times, image_white, permute(gains(:), [2,3,1]));

      max_val = max(image_tinted(:));
      image_tinted = image_tinted ./ max_val;

      image_down = image_tinted;
      illuminant_canonized = illuminant_tinted;
      % Because we've already applied the CCM, we force the rest of this
      % script to ignore the CCM, which can be done by setting it to a
      % identity matrix.
      CCM_project = eye(3);
    end

    max_val = max(image_down(:));
    image_down = min(1, image_down ./ max_val);

    if ~projects.(project).SENSOR.LINEAR_STATS
      image_down = ApplySrgbGamma(image_down);
    end

    % Convert the image to the specified bit depth, and cast accordingly.
    image_down = round((2^bit_width-1) * image_down);
    if bit_width <= 8
      image_down = uint8(image_down);
    elseif bit_width <= 16
      image_down = uint16(image_down);
    else
      assert(0);
    end

    if VISUALIZE
      try
        figure(1);
        % Visualize the image by white balancing it and applying the CCM and
        % gamma curve (if necessary).
        vis = double(image_down) / double(max(image_down(:)));
        if ~projects.(project).SENSOR.LINEAR_STATS
          vis = UndoSrgbGamma(vis);
        end
        vis = bsxfun(@times, vis, permute(1./illuminant_canonized(:), [2,3,1]));
        vis = reshape(reshape(vis, [], 3) * CCM_project', size(vis));
        vis = max(0, vis / max(vis(:)));
        vis = ApplySrgbGamma(vis);
        imagesc(vis); axis image off;
        title([num2str(i_file), ' - ', project])
        drawnow;
      catch me
        me;
      end
    end

    data.images.(project){end+1} = image_down;
    data.illuminants.(project){end+1} = illuminant_canonized;
    data.CCMs.(project){end+1} = CCM_project;
    data.camera_names.(project){end+1} = camera_name;
    data.exifs.(project){end+1} = exif;
  end
end

load(crossvalidationfolds_filename);

% Assert that the filenames are in the same order as is expected by the
% cross-validation struct.
roots = {Xfiles.name};
roots = cellfun(@(x) x(1:find(x=='.')-1), roots, 'UniformOutput', false);
for i_file = 1:length(data.filenames)
  filename = data.filenames{i_file};
  root_filename = filename( ...
    find(filename == '/', 1, 'last')+1 : find(filename == '.', 1, 'last')-1);
  assert(strcmp(root_filename, roots{i_file}))
end

% Turn the cross-validation test set splits into a single fold index per image.
is_test = [sparse(te_split{1}, 1, true, length(data.filenames), 1), ...
           sparse(te_split{2}, 1, true, length(data.filenames), 1), ...
           sparse(te_split{3}, 1, true, length(data.filenames), 1)];
assert(all(sum(is_test,2) == 1))
fold_idx = uint8(is_test * [1:3]');

% Write each project's images to disk.
for project = fieldnames(data.images)'
  project = project{1};
  fprintf('Writing %s', project);

  warning off all;
  output_folder = fullfile(gehlershi_folder, 'preprocessed', project);
  mkdir(output_folder);
  warning on all;

  % Write the cross-validation folds to the project, even though these
  % should be the same for all projects.
  crossvalidationfolds_filename = fullfile(output_folder, 'cvfolds.txt');
  fid = fopen(crossvalidationfolds_filename, 'w');
  for j = 1:length(fold_idx)
    fwrite(fid, num2str(fold_idx(j)));
    if j < length(fold_idx)
      fwrite(fid, sprintf('\n'));
    end
  end
  fclose(fid);

  % Write each image to disk, along with a *.txt text file that specifies the
  % illuminant and a *_ccm.txt text file that specifies the CCM.
  for i = 1:length(data.filenames)
    PrintDots(i, length(data.filenames), 72 - length(project))
    image_filename = fullfile(output_folder, num2str(i, '%06d.png'));
    imwrite(data.images.(project){i}, image_filename);

    illuminants_filename = fullfile(output_folder, num2str(i, '%06d.txt'));
    L = data.illuminants.(project){i};
    save(illuminants_filename, 'L', '-ascii', '-double')

    ccm_filename = fullfile(output_folder, num2str(i, '%06d_ccm.txt'));
    C = data.CCMs.(project){i};
    save(ccm_filename, 'C', '-ascii', '-double')

    camera_filename = fullfile(output_folder, num2str(i, '%06d_camera.txt'));
    fid = fopen(camera_filename, 'w');
    fprintf(fid, '%s', data.camera_names.(project){i});
    fclose(fid);

    exif_filename = fullfile(output_folder, num2str(i, '%06d_exif.mat'));
    exif = data.exifs.(project){i};
    save(exif_filename, 'exif');

    filename_filename = ...
      fullfile(output_folder, num2str(i, '%06d_filename.txt'));
    fid = fopen(filename_filename, 'w');
    fprintf(fid, '%s', data.filenames{i});
    fclose(fid);
  end
end
