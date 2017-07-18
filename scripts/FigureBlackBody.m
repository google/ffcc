clearvars;
addpath('../internal/')
addpath(genpath('../projects/'))

params = LoadProjectParams('GehlerShi');
load ~/tmp/awb_models/GehlerShi.mat
CCM = load('../projects/GehlerShi/tags/average_CCM.txt');

% Center the gain and bias (the filters are already centered).
bins = EnumerateBins(params);
shift = 33 - find(bins == 0);
model.B = circshift(model.B, [shift, shift]);
model.G = circshift(model.G, [shift, shift]);

params.HISTOGRAM.STARTING_UV = -1;

vis = @(x)ApplySrgbGamma( ...
  min(1, max(0, reshape(reshape(x, [], 3) * CCM', size(x)))));

B_min = min(model.B(:));
G_min = min(model.G(:));
B_vis = vis(VisualizeHistogram(model.B - B_min, params, 1));
G_vis = vis(VisualizeHistogram(model.G - G_min, params, 1));

n = size(model.F,1);
F_vis = {};
for i = 1:size(model.F,3)
  F_min = min(min(model.F(:,:,i)));
  F_vis{i} = vis( ...
    VisualizeHistogram(circshift(model.F(:,:,i), [n,n]/2) - F_min, params, 1));
end

imwrite(F_vis{1}, '../docs/figures/filter_pixel.png')
imwrite(F_vis{2}, '../docs/figures/filter_edge.png')
imwrite(B_vis, '../docs/figures/blackbody_bias.png')
imwrite(G_vis, '../docs/figures/blackbody_gain.png')
