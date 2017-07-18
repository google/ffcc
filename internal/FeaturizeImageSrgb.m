function [X, im_channels, im_linear] = FeaturizeImageSrgb(im_srgb, mask, params)
% Given an sRGB image and a binary mask indicating which pixels are to
% be trusted in that image, produces 2D histogram of the log-chroma of the
% image. Performs dithering to the sRGB image before linearizing it.

assert(isa(im_srgb, 'uint8'))

im_linear = UndoSrgbGamma(double(im_srgb) / 255);
im_linear = min(1, ...
  reshape(reshape(im_linear, [], 3) * inv(params.SENSOR.CCM)', ...
  size(im_srgb)));

assert(isa(im_linear, 'double'))
% TODO(yuntatsai): current C++ implementation assumes 8-bit input with
% fixed point math. This is due to the legacy from Links for performance
% reason. perhaps we should have ChannelizeImage all operates in
% fp32 [0..1] range, since we have much better performance headroom on
% Nexus.
im_linear = uint8(im_linear * 255);

[X, im_channels] = FeaturizeImage(im_linear, mask, params);
