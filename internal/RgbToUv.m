function uv = RgbToUv(rgb)
% Turns an RGB matrix (3xn) specifying the color of illuminants
% into a UV matrix (2xn) specifying the chroma of the white point of
% the scene (where the white point is 1/illuminant).

assert(size(rgb,1) == 3);
uv = [log(rgb(2,:) ./ rgb(1,:)); log(rgb(2,:) ./ rgb(3,:))];
