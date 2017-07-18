function im_edge = LocalAbsoluteDeviation(im)
% Compute Local Absolute Deviation in sliding window fashion. The window
% size is 3x3. If the input image is between [a, b] then the output image
% will be between [0, b-a].

% Upgrade to 16-bit because we have minus here
im_pad = Pad1(int16(im));

im_edge = {};
for c = 1:size(im,3)
  im_edge{c} = 0;
  for oi = -1:1
    for oj = -1:1
      if (oi == 0) && (oj == 0)
        continue
      end
      im_edge{c} = im_edge{c} + ...
                   abs(im_pad([1:size(im,1)] + oi + 1, ...
                              [1:size(im,2)] + oj + 1, c) - ...
                              int16(im(:,:,c)));
    end
  end
end
im_edge = cat(3, im_edge{:});
% Convert back to 8-bit
im_edge = uint8(bitshift(im_edge, -3));
