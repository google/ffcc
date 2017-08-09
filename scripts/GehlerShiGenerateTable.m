clearvars;

results = GehlerShiBaselines;

addpath(genpath('../projects/'));

extract_errors = @(x)[x.mean; x.median; x.tri; x.b25; x.w25];

[~, m] = GehlerShiHyperparams;
v = extract_errors(m.rgb_err);
results.metrics.fCCC = v(1:5)';
results.testtimes.fCCC = m.min_feature_time + m.min_filter_time;
results.traintimes.fCCC = mean(m.train_times);

[~, m] = GehlerShiDeepEHyperparams;
v = extract_errors(m.rgb_err);
results.metrics.fCCC_deepE = v(1:5)';
results.testtimes.fCCC_deepE = m.min_feature_time + m.min_filter_time;
results.traintimes.fCCC_deepE = mean(m.train_times);

[~, m] = GehlerShiThumbHyperparams;
v = extract_errors(m.rgb_err);
results.metrics.fCCC_thumb = v(1:5)';
results.testtimes.fCCC_thumb = m.min_feature_time + m.min_filter_time;
results.traintimes.fCCC_thumb = mean(m.train_times);

% Overriding test-time with numbers for Yun-Ta's Halide code,
% on a Intel Xeon CPU E5-2680.
results.testtimes.fCCC_thumb = 1.11 / 1000;

results.method_names.fCCC = ...
  '\texttt{J}) FFCC - full, 2 channels';
results.method_names.fCCC_deepE = ...
  '\texttt{O}) FFCC - full, 2 channels, +metadata';
results.method_names.fCCC_thumb = ...
  '\texttt{Q}) FFCC - thumb, 2 channels';

methods = fieldnames(results.metrics);

avg_errs = [];
sort_errs = [];
for i_method = 1:length(methods)
  method = methods{i_method};
  avg_errs(i_method) = geomean(results.metrics.(method));
  sort_errs(i_method) = geomean( ...
    results.metrics.(method)(~isnan(results.metrics.(method))));
end

x = cellfun(@(x) ~isempty(strfind(x, 'fCCC')), ...
  fieldnames(results.method_names), 'UniformOutput', false);
N_OURS = sum(cat(1, x{:}));

avg_errs_sort = sort_errs(1:end-N_OURS);
[~, sortidx] = sort(avg_errs_sort, 'descend');
sortidx = [sortidx, (length(sortidx) + 1) : length(avg_errs)];

metrics_best = inf(1,5);
avg_metrics_best = inf;
for field = fieldnames(results.metrics)'
  field = field{1};
  for i_metric = 1:5
    metrics_best(i_metric) = ...
      min(metrics_best(i_metric), results.metrics.(field)(i_metric));
  end
  avg_metrics_best = min(avg_metrics_best, geomean(results.metrics.(field)));
end

fid = fopen('gehler_shi_table.tex', 'w');
for i_method = sortidx
  method = methods{i_method};
  fprintf(fid, '%s & ', results.method_names.(method));
  for i_metric = 1:length(results.metrics.(method))
    if metrics_best(i_metric) == results.metrics.(method)(i_metric)
      fprintf(fid, ' \\cellcolor{Yellow} ');
    end
    if isnan(results.metrics.(method)(i_metric))
      fprintf(fid, '- & ');
    else
      fprintf(fid, '$ %0.2f $ & ', results.metrics.(method)(i_metric));
    end
  end
  if geomean(results.metrics.(method)) == avg_metrics_best
    fprintf(fid, ' \\cellcolor{Yellow} ');
  end
  if isnan(geomean(results.metrics.(method)))
    fprintf(fid, '- & ');
  else
    fprintf(fid, '$ %0.2f $ & ', geomean(results.metrics.(method)));
  end
  if isfield(results.testtimes, method) && ~isnan(results.testtimes.(method))
    t = results.testtimes.(method);
    fprintf(fid, ['$ %0.', num2str(max(0, ceil(-log10(t)) + 1)), 'f $ & '], ...
      results.testtimes.(method));
  else
    fprintf(fid, ' - & ');
  end
  if isfield(results.traintimes, method)
    if isnan(results.traintimes.(method))
      fprintf(fid, ' - ');
    else
      fprintf(fid, '$ %d $', round(results.traintimes.(method)));
    end
  else
    fprintf(fid, ' - ');
  end

  if i_method ~= sortidx(end)
    fprintf(fid, '\\\\\n', geomean(results.metrics.(method)));
  end

  if (i_method == sortidx(end-N_OURS)) || (i_method == sortidx(end-1))
    fprintf(fid, '\\hline\n');
  end
end
fprintf(fid, '\n');
fclose(fid);
