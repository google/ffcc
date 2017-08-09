% Precomputes a feature vector for images in the Gehler-Shi dataset, using the
% exif and camera name as input.

clearvars;
addpath('../internal/')

paths = DataPaths;
folder = fullfile(paths.gehler_shi, 'preprocessed', 'GehlerShi');

dirents = dir(fullfile(folder, '*.png'));
image_filenames = ...
  cellfun(@(x)fullfile(folder, x), {dirents.name}, 'UniformOutput', false);

for i_file = 1:length(image_filenames)
  PrintDots(i_file, length(image_filenames))
  image_filename = image_filenames{i_file};

  camera_filename = [image_filename(1:end-4), '_camera.txt'];
  fid = fopen(camera_filename, 'r');
  sensor_name = strtrim(fgetl(fid));
  fclose(fid);

  exif_filename = [image_filename(1:end-4), '_exif.mat'];
  exif = getfield(load(exif_filename), 'exif');
  feature_vec = [log(exif.Shutter_Speed*64); log(exif.F_Number/6); 1];

  flag = strcmp(sensor_name, 'Canon1D');
  feature_vec = [feature_vec; feature_vec * flag; feature_vec * ~flag];

  feature_filename = [image_filename(1:end-4), '_exif.txt'];
  save(feature_filename, 'feature_vec', '-ascii', '-double')
end
