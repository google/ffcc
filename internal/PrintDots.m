% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

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

