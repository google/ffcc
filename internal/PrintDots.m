function PrintDots(iter, n_iters, n_print)
% When called within a for loop where iter ranges from 1:n_iters, causes
% periods to be printed to simulate a progress bar, with n_print (default = 80)
% periods printed in total.

if nargin < 3
  n_print = 80;
end

x = round(1:((n_iters-1)/(n_print-1)):n_iters);
if isempty(x)
  fprintf(repmat('.', [1, n_print]));
  fprintf('\n');
else
  fprintf(repmat('.', [1, sum(x == iter)]));
  if iter == x(end)
    fprintf('\n');
  end
end

