function im_channels = ChannelizeImage(im, mask)
% Generate feature images (color and gradient) and combine them into
% different cell channels.

assert(isa(mask, 'logical'));

im_channels = {};

im_channels{1} = cast(bsxfun(@times, double(im), mask), 'like', im);
im_channels{2} = cast(MaskedLocalAbsoluteDeviation(im, mask), 'like', im);
