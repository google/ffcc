clearvars

model = getfield(load('~/tmp/awb_models/ChengCanon1DsMkIII.mat'), 'model');

addpath(genpath('../'));

paths = DataPaths;

params = LoadProjectParams('ChengCanon1DsMkIII');

params_big = params;
params_big.HISTOGRAM.NUM_BINS = 3 * params.HISTOGRAM.NUM_BINS;
params_big.HISTOGRAM.STARTING_UV = params.HISTOGRAM.STARTING_UV ...
  - params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE;

% image_indices = [129 131 183 213 214 219];
image_index = 129;

I = imread([fullfile(paths.cheng, 'preprocessed/Cheng/Canon1DsMkIII/'), ...
  num2str(image_index, '%06d'), '.png']);
if image_index == 129
  I = I(:,end-256:end,:);
else
  I = I(:,[1:256]+32,:);
end
I = double(I) / double(max(I(:)));

L = load([fullfile(paths.cheng, 'preprocessed/Cheng/Canon1DsMkIII/'), ...
  num2str(image_index, '%06d'), '.txt']);
CCM = load([fullfile(paths.cheng, 'preprocessed/Cheng/Canon1DsMkIII/'), ...
  num2str(image_index, '%06d'), '_ccm.txt']);

W = min(1, bsxfun(@times, I, permute(L(2) ./ L(:), [2,3,1])));

vis = @(x)( ...
  ApplySrgbGamma(max(0, min(1, reshape((reshape(x, [], 3) * CCM'), size(x))))));

X = FeaturizeImage(I, ~any(I == 0, 3), params);
X_big = FeaturizeImage(I, ~any(I == 0, 3), params_big);

uv = RgbToUv(1./L);
uv = round(uv / params.HISTOGRAM.BIN_SIZE) * params.HISTOGRAM.BIN_SIZE;
gains = UvToRgb(uv);
gains = gains / gains(2);
gains = gains * min(1, 2 / max(gains));

I_ = bsxfun(@times, permute(gains(:), [2,3,1]), double(I));
I_ = I_ ./ max(I_(:));
X_ = FeaturizeImage(I_, [], params);

di = round(uv(1) / params.HISTOGRAM.BIN_SIZE);
dj = round(uv(2) / params.HISTOGRAM.BIN_SIZE);

X_shift = circshift(X, [di, dj]);

assert(corr(X_(:), X_shift(:)) > 0.99)

Y = -uv;
[state, metadata] = EvaluateModel( ...
  model.F_fft, model.B, X, fft2(X), Y, [], [], params);

P_dealiased = RenderHistogramGaussian(state.mu, state.Sigma, [], ...
  3*params.HISTOGRAM.NUM_BINS, false, params);
P_dealiased = vis(double(P_dealiased)/255);

width = params.HISTOGRAM.NUM_BINS * params.HISTOGRAM.BIN_SIZE;
P_aliased = {};
for oi = -1:1
  for oj = -1:1
    params_ = params;
    P_vis = RenderHistogramGaussian(state.mu + [width*oi; width*oj], ...
      state.Sigma, [], [], false, params, width*oi, width*oj);
    P_vis = vis(double(P_vis)/255);
    P_aliased{oi+2}{oj+2} = P_vis;
  end
  P_aliased{oi+2} = cat(2, P_aliased{oi+2}{:});
end
P_aliased = cat(1, P_aliased{:});

mult = 2;
hist_aliased = vis(VisualizeHistogram(repmat(X(:,:,1), [3,3]), params_big));
ii = (size(hist_aliased,1)/3) + [1:(size(hist_aliased,1)/3)];
jj = (size(hist_aliased,1)/3) + [1:(size(hist_aliased,1)/3)];
hist_aliased(ii, jj, :) = hist_aliased(ii, jj, :)*mult;
hist_aliased = hist_aliased / mult;

mult = 1.25;
ii = (size(P_aliased,1)/3) + [1:(size(P_aliased,1)/3)];
jj = (size(P_aliased,1)/3) + [1:(size(P_aliased,1)/3)];
P_aliased(ii, jj, :) = P_aliased(ii, jj, :)*mult;
P_aliased = P_aliased / mult;


params_filt = params;
params_filt.HISTOGRAM.STARTING_UV = ...
  -(params_filt.HISTOGRAM.NUM_BINS/2 - 0.5) * params_filt.HISTOGRAM.BIN_SIZE;

filt = fftshift(model.F(:,:,1));
filt = filt - min(filt(:));
filt_vis = vis(VisualizeHistogram(filt, params_filt, 2));
filt_vis_pad = ones(size(filt_vis, 1) * 3, size(filt_vis, 1) * 3, 3);
ii = size(filt_vis, 1) + [1:size(filt_vis, 1)];
jj = size(filt_vis, 2) + [1:size(filt_vis, 2)];
filt_vis_pad(ii, jj, :) = filt_vis;

filt2 = fftshift(model.F(:,:,2));
filt2 = filt2 - min(filt2(:));
filt2_vis = vis(VisualizeHistogram(filt2, params_filt, 2));
filt2_vis_pad = ones(size(filt2_vis, 1) * 3, size(filt2_vis, 1) * 3, 3);
ii = size(filt2_vis, 1) + [1:size(filt2_vis, 1)];
jj = size(filt2_vis, 2) + [1:size(filt2_vis, 2)];
filt2_vis_pad(ii, jj, :) = filt2_vis;

bias = model.B(:,:,1);
bias = bias - min(bias(:));
bias_vis = vis(VisualizeHistogram(bias, params_filt, 1));
bias_vis_pad = ones(size(bias_vis, 1) * 3, size(bias_vis, 1) * 3, 3);
ii = size(bias_vis, 1) + [1:size(bias_vis, 1)];
jj = size(bias_vis, 2) + [1:size(bias_vis, 2)];
bias_vis_pad(ii, jj, :) = bias_vis;

mult = 2;
prob_aliased = ...
  vis(VisualizeHistogram(repmat(metadata.P(:,:,1), [3,3]), params_big));
ii = (size(prob_aliased,1)/3) + [1:(size(prob_aliased,1)/3)];
jj = (size(prob_aliased,1)/3) + [1:(size(prob_aliased,1)/3)];
prob_aliased(ii, jj, :) = prob_aliased(ii, jj, :)*mult;
prob_aliased = prob_aliased / mult;

imwrite(vis(I), '../docs/figures/overview_input.png');
imwrite(vis(VisualizeHistogram(X_big(:,:,1), params_big)), ...
  '../docs/figures/overview_histogram_big.png');
imwrite(vis(VisualizeHistogram(X(:,:,1), params)), ...
  '../docs/figures/overview_histogram_aliased_crop.png');
imwrite(hist_aliased, '../docs/figures/overview_histogram_aliased.png');
imwrite(P_dealiased, '../docs/figures/overview_prediction_dealiased.png');
imwrite(P_aliased, '../docs/figures/overview_prediction_aliased.png');
imwrite(vis(W), '../docs/figures/overview_prediction_output.png');

imwrite(filt_vis, '../docs/figures/overview_filter.png');
imwrite(filt_vis_pad, '../docs/figures/overview_filter_padded.png');
imwrite(bias_vis, '../docs/figures/overview_bias.png');
imwrite(bias_vis_pad, '../docs/figures/overview_bias_padded.png');
imwrite(prob_aliased, '../docs/figures/overview_prob_aliased.png');
