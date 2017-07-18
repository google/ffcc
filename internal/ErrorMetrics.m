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
