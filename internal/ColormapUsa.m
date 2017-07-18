function output = ColormapUsa
% Produces a colormap that goes blue-white-red. Useful for visualizing
% images where negative, positive, and zero values have qualitatively
% different meanings.
%
% If no output arguments are specified, the current figure's colormap is
% set. If an output argument is specified, then the colormap is returned
% and the figure is not modified.
%
% Usage example:
%   imagesc(image, [-1, 1]); ColormapUsa;

n = 512;
x = [0:n]'/n;

b = 1 + min(0, 1 - 2*x);
r = 1 + min(0, 1 - 2*(1-x));
cmap = [r, min(r, b), b];

if nargout == 0
  colormap(cmap);
else
  output = cmap;
end
