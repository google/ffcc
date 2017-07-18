clearvars;
addpath('../internal/')
I = imread('jemison.jpg');
I = I(1:1700,[1:1700] + 350,:);
I = imresize(I, [256, nan]);
I = double(I)/255;
I = I - min(I(:));
I = I ./ max(I(:));

p1 = [38, 95];
p2 = [123, 176];
mask = false(size(I));
mask([p1(1) + [-1:1], p2(1) + [-1:1]], (p1(2)-1):(p2(2)+1), :) = true;
mask((p1(1)-1):(p2(1)+1), [p1(2) + [-1:1], p2(2) + [-1:1]], :) = true;

B = I;
B(mask) = nan;

n = 128;
numer = 0;
denom = 0;
D = zeros(size(B(:,:,1)));
for i = 1:(size(B,1)/n)
  for j = 1:(size(B,1)/n)
    B_sub = B((i-1)*n + [1:n], (j-1)*n + [1:n], :);
    numer = numer + B_sub;
    denom = denom + 1;
    D((i-1)*n + [1:n], (j-1)*n + [1:n]) = denom;
  end
end
B = numer / denom;

A = I;
A(mask) = nan;

col = [1, 1, 0];
for c = 1:3
  Ac = A(:,:,c);
  Bc = B(:,:,c);
  Ac(isnan(Ac(:,:))) = col(c);
  Bc(isnan(Bc(:,:))) = col(c);
  A(:,:,c) = Ac;
  B(:,:,c) = Bc;
end

B_rep = padarray(B, (size(A,1)-size(B,1))/2 * [1,1], 1, 'both');

imwrite(A, '../docs/figures/face1.png')
imwrite(B, '../docs/figures/face2.png')
imwrite(B_rep, '../docs/figures/face2_rep.png')
