% Precompute Starburst features for the Gehler-Shi dataset.

clearvars;

addpath('../internal/')
binary_path = '../../../../../../blaze-bin/image/understanding/tensorflow/client/model_runner_example';

paths = DataPaths;
folder = paths.gehler_shi;

dirents = dir(fullfile(folder, '*.png'));
data = {};
for i_file  = 1:length(dirents)
  fprintf('%d ', i_file)

  root = dirents(i_file).name;
  root = root(1:(find(root == '.', 1, 'first')-1));

  im_filename = fullfile(folder, dirents(i_file).name);

  im = double(imread(im_filename)) / double(intmax('uint16'));
  % Scale the G channel by 0.6 to very-roughly white balance the image.
  im = bsxfun(@times, im, cat(3, 1, 0.6, 1));
  im = im ./ max(im(:));

  ccm = load(fullfile(folder, [root, '_ccm.txt']));

  im = uint8(round(255 * ApplySrgbGamma( ...
    reshape(max(0, min(1, reshape(im, [], 3) * ccm')), size(im)))));

  % Crop the center.
  min_sz = min(size(im,1), size(im,2));
  i = size(im,1)/2 + [-min_sz/2+1:min_sz/2];
  j = size(im,2)/2 + [-min_sz/2+1:min_sz/2];
  im_sq = im(i, j, :);

  im_224 = imresize(im_sq, [224, 224]);

  im_filename = '/tmp/im.jpg';
  tmp_filename = '/tmp/feature.txt';
  imwrite(im_224, im_filename, 'Quality', 95);
  system([binary_path, ' --input_file=', im_filename, ...
    ' --image_input_type=dist_belief.Image', ...
    ' --tf_graph_file=/usr/local/google/home/barron/data/starburst/tf_graph_starburst.pbtxt', ...
    ' --tf_checkpoint_file=/usr/local/google/home/barron/data/starburst/2015-09-01_ps-tensor_slices-ckpt_00000001-p0000-of-0001', ...
    ' --input_node=input --output_node=embedding-norm-concat > ', ...
    tmp_filename]);
  fprintf('raw complete\n');

  feature_vec = [];
  fid = fopen(tmp_filename, 'r');
  while true
  line = fgetl(fid);
  if line(1) == '#'
    continue;
  end
  if line == -1
    break
  end
  feature_vec = [feature_vec; reshape(str2num(line), [], 1)];
  end

  feat_filename = fullfile(folder, [root, '_starburst_raw.txt']);
  save(feat_filename, 'feature_vec', '-ascii', '-double')
end
