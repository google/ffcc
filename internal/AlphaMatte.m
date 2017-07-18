function [rgbC, alphaC] = AlphaMatte(rgbA, alphaA, rgbB, alphaB)
% Apply an "over" operator, where (rgbA, alphaA) is overlayed on (rgbB, alphaB).
% The output is a composite image and alpha matte (rgbC, alphaC).

rgbC = bsxfun(@rdivide, ...
  bsxfun(@times, rgbA, alphaA) + bsxfun(@times, rgbB, alphaB .* (1 - alphaA)), ...
  alphaA + alphaB .* (1 - alphaA));
alphaC = alphaA + alphaB .* (1 - alphaA);
