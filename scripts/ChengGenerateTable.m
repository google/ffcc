clearvars;

addpath('../internal/');

results = ChengBaselines;

CAMERA_NAMES = ...
  {'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200', ...
   'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'};

addpath(genpath('../projects/'));
error_names = fieldnames(results.Canon1DsMkIII);

results.methods{end+1} = 'CCC';
results.method_names.CCC = 'CCC \cite{BarronICCV2015}';
for i_camera = 1:length(CAMERA_NAMES)
  camera_name = CAMERA_NAMES{i_camera};
  ccc_avg_err.mean = 2.38;
  ccc_avg_err.median = 1.48;
  ccc_avg_err.tri = 1.692;
  ccc_avg_err.b25 = 0.452;
  ccc_avg_err.w25 = 5.852;
  for i_error = 1:length(error_names)
    error_name = error_names{i_error};
    results.(camera_name).(error_name)(end+1) = ccc_avg_err.(error_name);
  end
end

results.methods{end+1} = 'fCCC_thumb';
results.methods{end+1} = 'fCCC_full';

results.method_names.fCCC_thumb = '\texttt{Q}) FFCC - thumb, 2 channels';
results.method_names.fCCC_full = '\texttt{M}) FFCC - full, 4 channels';

hyperparams_fieldnames = { ...
  'PRETRAIN_MULTIPLIER', ...
  'LIKELIHOOD_MULTIPLIER', ...
  'FILTER_MULTIPLIERS', ...
  'BIAS_MULTIPLIER', ...
  'LOGGAIN_MULTIPLIER', ...
  'FILTER_SHIFTS', ...
  'BIAS_SHIFT', ...
  'LOGGAIN_SHIFT'};

full_logsum_hyp = struct();
thumb_logsum_hyp = struct();
for i_field = 1:length(hyperparams_fieldnames)
  field = hyperparams_fieldnames{i_field};
  eval(['thumb_logsum_hyp.HYPERPARAMS.', field, ' = 0;'])
  eval(['full_logsum_hyp.HYPERPARAMS.', field, ' = 0;'])
end

for i_camera = 1:length(CAMERA_NAMES)
  camera_name = CAMERA_NAMES{i_camera};
  [thumb_hyp, thumb_metrics] = eval(['Cheng', camera_name, 'ThumbHyperparams']);
  [full_hyp, full_metrics] = eval(['Cheng', camera_name, 'Hyperparams']);

  for i_error = 1:length(error_names)
    error_name = error_names{i_error};
    results.(camera_name).(error_name)(end+1) = ...
      thumb_metrics.rgb_err.(error_name);
    results.(camera_name).(error_name)(end+1) = ...
      full_metrics.rgb_err.(error_name);
  end
end

avg_error = struct();
for i_error = 1:length(error_names)
  error_name = error_names{i_error};
  total_log_error = 0;
  for i_camera = 1:length(CAMERA_NAMES)
    total_log_error = total_log_error + log(results.(camera_name).(error_name));
  end
  avg_error.(error_name) = exp(total_log_error / length(CAMERA_NAMES));
end

total_log_error = 0;
for i_error = 1:length(error_names)
  error_name = error_names{i_error};
  total_log_error = total_log_error + log(avg_error.(error_name));
end
avg_error.avg = exp(total_log_error / length(error_names));

[sorted, sortidx] = sort(avg_error.avg(1:end-2), 'descend');
sortidx = [sortidx, length(avg_error.avg) + [0, -1]];

fid = fopen('./cheng_table.tex', 'w');
for i_method = sortidx
  if (i_method == (sortidx(end-1))) || (i_method == (sortidx(end)))
    fprintf(fid, '\\hline\n');
  end

  method = results.methods{i_method};
  fprintf(fid, '%s & ', results.method_names.(method));
  for i_error = 1:length(error_names)
    error_name = error_names{i_error};
    error = avg_error.(error_name)(i_method);
    if avg_error.(error_name)(i_method) == min(avg_error.(error_name))
      fprintf(fid, ' \\cellcolor{Yellow} ');
    end
    fprintf(fid, '$ %0.2f $ & ', error);
  end

  error_name = 'avg';
  if avg_error.(error_name)(i_method) == min(avg_error.(error_name))
    fprintf(fid, ' \\cellcolor{Yellow} ');
  end
  fprintf(fid, '$ %0.2f $', avg_error.avg(i_method));

  if i_method ~= sortidx(end)
    fprintf(fid, ' \\\\\n');
  end
end
fprintf(fid, '\n');
fclose(fid);
