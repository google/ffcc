function PrintCropped(filename, res)

print('-dpng', filename, ['-r', num2str(res)]);
im = imread(filename);

W = all(im == 255,3);
idx1 = find(~all(W,2), 1, 'first') : find(~all(W,2), 1, 'last');
idx2 = find(~all(W,1), 1, 'first') : find(~all(W,1), 1, 'last');
im = im(idx1, idx2,:);
imwrite(im, filename);
