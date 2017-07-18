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

function metrics = ErrorMetrics(errors)
% Takes as input a vector of errors, and returns a struct of surface statistics
% for those errors. These metrics are:
%    mean   = The arithmetic mean of the errors.
%    mean2  = The root-mean-squared of the errors.
%    mean4  = The quartic root of the mean quartic error.
%    median = The median error.
%    tri    = The trimean of the errors.
%    b25    = the mean of the bottom quartile of the errors (not the same as the
%             25th percentile)
%    w25    = the mean of the top quartile of the data (not the same as the 75th
%             percentile)
%    w05    = the mean of the top 5% of the data (not the same as the 95th
%             percentile).
%    max    = the maximum error.

percentiles = prctile(errors, [25, 50, 75, 95]);
metrics.mean = mean(errors);
metrics.mean2 = sqrt(mean(errors.^2));
metrics.mean4 = mean(errors.^4).^(1/4);
metrics.median = percentiles(2);
metrics.tri = (percentiles(1:3) * [1;2;1])/4;
metrics.b25 = mean(errors(errors <= percentiles(1)));
metrics.w25 = mean(errors(errors >= percentiles(3)));
metrics.w05 = mean(errors(errors >= percentiles(4)));
metrics.max = max(errors);
