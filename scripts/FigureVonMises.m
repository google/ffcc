clearvars;
addpath('../internal')
addpath(genpath('../projects'))

project = 'GehlerShiSmall';
folder = ['~/tmp/awb_vis/', project];
params = LoadProjectParams(project);
dirents = dir(fullfile(folder, '*_P.mat'));

output_folder = '~/tmp/vis_vonmises/';
mkdir(output_folder)

Ps = {};
for i_file = 1:length(dirents)
  load(fullfile(folder, dirents(i_file).name));
  Ps{i_file} = P;

  n = size(P,1);
  i = [0:(n-1)]';
  j = [0:(n-1)];
  mu = [sum(sum(bsxfun(@times, i, P))), sum(sum(bsxfun(@times, j, P)))]';
  P1 = sum(P,2);
  P2 = sum(P,1)';

  bins = [0:(size(P,1)-1)]';
  wrapped1 = bins - round(mu(1));
  wrapped2 = bins - round(mu(2));

  E1 = sum(P1 .* wrapped1);
  E2 = sum(P2 .* wrapped2);
  Sigma1 = sum(P1 .* wrapped1.^2) - E1.^2;
  Sigma2 = sum(P2 .* wrapped2.^2) - E2.^2;
  Sigma12 = sum(sum(P .* bsxfun(@times, wrapped1, wrapped2'))) - E1 * E2;

  Sigma = [Sigma1, Sigma12; ...
           Sigma12, Sigma2];
   mu = mu + 1;

   [mu_, Sigma_] = FitBivariateVonMises(P);

   m = 1024;
   i = imresize([0:(n+1)], m/n);
   i = i(1,(m/n+1):(end-(m/n)));
   [ii,jj] = ndgrid(i, i);

   Pv = imresize(P, [m, m], 'nearest');

   k = 65;
   plus = false(k,k);
   plus( (k+1)/2, :) = true;
   plus( :, (k+1)/2) = true;
   ecks = false(k,k);
   [ki,kj] = ndgrid(1:k, 1:k);
   ecks(ki == kj) = true;
   ecks(ki == (k - kj+1)) = true;

   mahal_dist = inf;
   for oi = -1:1
     for oj = -1:1
       X = bsxfun(@minus, [ii(:), jj(:)]', [oi;oj]*n);
       Xc = bsxfun(@minus, X, mu_);
       iSigma = inv(Sigma_);
       Xc_iSigma = iSigma * Xc;
       mahal_dist = min(mahal_dist, reshape(sum(Xc .* Xc_iSigma,1), size(ii)));
     end
   end
   mask = mahal_dist <= 3;
   f_plus = [0, 1, 0; 1, 1, 1; 0, 1, 0];
   n_dilate = 8;

   prediction_ = ((conv2(double(mask), f_plus, 'same') > 0) & ~mask) | ...
     conv2(double(mahal_dist == min(mahal_dist(:))), double(plus), 'same') > 0;
   for i_dilate = 1:n_dilate
     prediction_ = conv2(double(prediction_), f_plus, 'same') > 0;
   end

   X = [ii(:), jj(:)]';
   Xc = bsxfun(@minus, X, mu);
   iSigma = inv(Sigma);
   Xc_iSigma = iSigma * Xc;
   mahal_dist = reshape(sum(Xc .* Xc_iSigma,1), size(ii));
   mask = mahal_dist <= 3;
   prediction = ((conv2(double(mask), f_plus, 'same') > 0) & ~mask);

   idx = find(prediction);
   i = ii(idx);
   j = jj(idx);
   ij = [i - mu(1), j - mu(2)] * sqrtm(inv(Sigma));
   [th, r] = cart2pol(ij(:,1), ij(:,2));
   prediction(prediction) = mod(th / 0.125, 2) > 1;

   prediction = prediction | ...
     conv2(double(mahal_dist == min(mahal_dist(:))), double(ecks), 'same') > 0;
   for i_dilate = 1:n_dilate
     prediction = conv2(double(prediction), f_plus, 'same') > 0;
   end

   Pv = Pv - min(Pv(:));
   Pv = Pv ./ max(Pv(:));
   vis = repmat(1-sqrt(Pv), [1,1,3]);

   M = prediction | prediction_;
   for i_feather = 1:9
     M = conv2(double(M), f_plus, 'same') > 0;
     vis(repmat(M, [1,1,3])) = 1-0.8*(1-vis(repmat(M, [1,1,3])));
   end

   vis1 = vis(:,:,1);
   vis2 = vis(:,:,2);
   vis3 = vis(:,:,3);

   vis1(prediction) = 1;
   vis2 = vis2 .* (1-prediction);
   vis3 = vis3 .* (1-prediction);

   vis1 = vis1 .* (1-prediction_);
   vis2 = vis2 .* (1-prediction_);
   vis3(prediction_) = 1;

   vis = cat(3, vis1, vis2, vis3);
   vis = imresize(vis, 0.25, 'bilinear');

   imagesc(vis); axis image off; drawnow;

   imwrite(vis, fullfile(output_folder, num2str(i_file, '%04d.png')));
end
