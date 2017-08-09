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

function [image, mask, camera_name, illuminant, CCM, exif] = ...
  ReadImage(image_filename)
% A helper function for reading images from the "Gehler Shi" or "Cheng"
% datasets, without having to worry about each camera's black levels or
% saturation values. This function will only work on images being loaded from
% these two dataset, formatting as they are on the Gcam RAID.
% Outputs:
%   image - A (linear RGB) image in [0, 1], after black-level subtraction.
%   mask - A binary mask which is true iff that pixel is not saturated or within
%     the color checker (ie, trustworthy in AWB).
%   camera_name - The name of the camera/sensor that the image is from.
%   illuminant - The ground-truth RGB gains for the image.
%   CCM - The color correction matrix for the image.

% A hardcoded scalar that determines what fraction of the saturation value we
% should actually consider saturation to happen at. Setting this to values <
% 1 produces slightly more conservative saturation masking.

paths = DataPaths;

SATURATION_SCALE = 0.95;

exif = struct();

image = double(imread(image_filename));

% If the image appears to be in the Gehler-Shi dataset.
if strfind(image_filename, '/shi_gehler/')

  % The camera for this dataset can be determined from the filename.
  if strcmp( ...
      image_filename(find(image_filename == '/', 1, 'last') + [1:3]), 'IMG')
    camera_name = 'Canon5D';
  else
    camera_name = 'Canon1D';
  end

  % Constants for this dataset are defined externally.
  % TODO(barron): consider dropping this file and handling CCM loading
  % differently.
  gehlershi_sensor = GehlerShiSensor;

  black_level = gehlershi_sensor.BLACK_LEVEL.(camera_name);
  saturation = gehlershi_sensor.SATURATION.(camera_name);
  CCM = gehlershi_sensor.CCMs.(camera_name);

  gehlershi_folder = paths.gehler_shi;
  root_filename = ...
    image_filename((find(image_filename == '/', 1, 'last') + 1) : end-4);

  % To find the illuminant we scrape all filenames in the dataset and find the
  % index which corresponds to the input filename, and then use that index to
  % find the illuminant.
  % The creators of this dataset chose to store ground-truth illuminants in a
  % .mat file which is indexed on the order of the image in its folder, and so
  % this code may break if run on a filesystem which sorts images arbitrarily.
  illuminants_filename = ...
    fullfile(gehlershi_folder, 'real_illum_568.mat');
  illuminants = load(illuminants_filename);
  dirents = dir(fullfile(paths.gehler_shi, 'images'));
  dirents = {dirents.name};
  dirents = dirents(cellfun(@(x)(x(1) ~= '.'), dirents));
  assert(length(dirents) == size(illuminants.real_rgb,1));
  filename = image_filename((find(image_filename == '/', 1, 'last') + 1) : end);
  i_file = find(cellfun(@(x) strcmp(x, filename), dirents));
  assert(length(i_file) == 1)
  illuminant = double(illuminants.real_rgb(i_file,:));
  illuminant = illuminant(:) ./ sqrt(sum(illuminant.^2));

  % The mask for the color checker is encoded as the (weirdly scaled)
  % coordinates of the corners of the color checker.
  coordinates_folder = fullfile(gehlershi_folder, 'coordinates/');
  cc_coord = load(fullfile(coordinates_folder, [root_filename '_macbeth.txt']));
  scale = cc_coord(1, [2 1]) ./ [size(image,1) size(image,2)];
  mask = ~roipoly(image, ...
    cc_coord([2 4 5 3],1) / scale(1), cc_coord([2 4 5 3],2) / scale(2));

  % Load in the EXIF data as a struct.
  idx = strfind(image_filename, '/images/');
  raw_glob = ...
    fullfile(image_filename(1:idx), 'RAW', [image_filename(idx+8:end-3), '*']);
  raw_filename = getfield(dir(raw_glob), 'name');
  raw_path = ...
    fullfile(raw_glob(1:find(raw_glob == '/', 1, 'last')), raw_filename);
  system(fullfile(['exiftool ', raw_path, ' > /tmp/exif.txt']));
  fid = fopen('/tmp/exif.txt', 'r');
  while true
    line = fgetl(fid);
    if line == -1
      break
    end
    field = strtrim(line(1:32));
    field( ...
      (field == ' ') | (field == '/') | (field == '-') | (field == '.')) ...
      = '_';
    value = strtrim(line(34:end));
    try
      value = eval(value);
    catch me
      me;
    end
    if ~isnan(str2double(value))
      value = str2double(value);
    end
    exif.(field) = value;
  end
  fclose(fid);

elseif strfind(image_filename, '/cheng/')

  cheng_folder = paths.cheng;

  root_filename = ...
    image_filename((find(image_filename == '/', 1, 'last') + 1) : end-4);

  % The name of the camera can be scraped directly from the folder containing
  % the image.
  camera_name = image_filename((strfind(image_filename, '/cheng/') + 7) : end);
  camera_name = camera_name(1:(find(camera_name == '/', 1, 'first')-1));

  % Each camera folder contains a .mat file with the ground-truth illuminants
  % and sensor properties.
  gt = load(fullfile(cheng_folder, camera_name, [camera_name, '_gt.mat']));

  black_level = gt.darkness_level;
  saturation = gt.saturation_level;

  % To find the illuminant we scrape all filenames for this camera and find the
  % index which corresponds to the input filename, and then use that index to
  % find the illuminant.
  % The creators of this dataset chose to store ground-truth illuminants in a
  % .mat file which is indexed on the order of the image in its folder, and so
  % this code may break if run on a filesystem which sorts images arbitrarily.
  i_file = cellfun(@(x) ~isempty(strfind(root_filename, x)), ...
    gt.all_image_names, 'UniformOutput', false);
  i_file = cat(1, i_file{:});
  illuminant = gt.groundtruth_illuminants(i_file,:);
  illuminant = illuminant(:) ./ sqrt(sum(illuminant.^2));

  % The color checker mask is stored as the coordinates of a rectangle.
  mask = true(size(image,1), size(image,2));
  mask(gt.CC_coords(i_file,1) : gt.CC_coords(i_file,2), ...
       gt.CC_coords(i_file,3) : gt.CC_coords(i_file,4), :) = false;

  % We can load a precomputed CCM for this camera from within google3. Some
  % elaborate string-hacky is necessary to deal with the fact that this
  % function can be called from many folders.
  % TODO(barron): make this cleaner.
  working_dir = pwd;
  CCM = load(fullfile(working_dir(1:strfind(working_dir, 'matlab')+5), ...
    'projects', 'Cheng', 'tags', [camera_name,'_CCM.txt']));

else
  assert(0)
end

% Check that the CCM does not move the white point.
assert(max(abs(sum(CCM, 2) - 1)) < 1e-12)

% Subtract out the black level and adjust the saturation value accordingly.
image = max(image-black_level,0);
saturation = saturation - black_level;

% Scale and clamp the image.
image = min(1, image / (SATURATION_SCALE * saturation));

% Update the mask to ignore saturated pixels.
mask = mask & all(image < 1, 3);
