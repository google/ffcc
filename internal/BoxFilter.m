function Y = BoxFilter(X, r)
% Applies a box filter of half-width r to input image X.
% This code only exists to support ChannelizeImage2015, which is legacy code
% included for comparison and paper-writing.

X_sum = cumsum(cumsum( ...
  [zeros(1, size(X,2)+1, size(X,3)); zeros(size(X,1), 1, size(X,3)), X], 2), 1);

i_lo = max(1, (1-r) : (size(X_sum,1)-r-1));
i_hi = min(size(X_sum,1), (2+r) : (size(X_sum,1)+r));
j_lo = max(1, (1-r) : (size(X_sum,2)-r-1));
j_hi = min(size(X_sum,2), (2+r) : (size(X_sum,2)+r));

Y = (X_sum(i_lo, j_lo,:) ...
   + X_sum(i_hi, j_hi,:) ...
   - X_sum(i_lo, j_hi,:) ...
   - X_sum(i_hi, j_lo,:));
