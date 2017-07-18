clearvars;

folder = '/usr/local/google/home/barron/tmp/awb_vis/GehlerShi2015/';

n = 568;
err = [];
conf = [];
for i = 1:n
  err(i) = load(fullfile(folder, [num2str(i, '%08d'), '_error.txt']));
  conf(i) = load(fullfile(folder, [num2str(i, '%08d'), '_confidence.txt']));
end
err = err(:);
conf = conf(:);
entropy = -6.9315 - log(conf);
rank = ([1:length(err)] - 0.5) / length(err);

[~, sortidx] = sort(entropy, 1, 'ascend');
err_curve = cumsum(err(sortidx)) / length(err);
err_baseline = ([1:length(err)] - 0.5) / length(err) * mean(err);
[~, oracleidx] = sort(err, 1, 'ascend');
err_oracle = cumsum(err(oracleidx)) / length(err);

assert(abs(mean(err_baseline)*2 - mean(err)) < 1e-10)
assert(abs(err_curve(end) - mean(err)) < 1e-10);

fprintf('Gehler: err = %0.3f / %0.3f\n', ...
  2*mean(err_curve), 2*mean(err_baseline))

figure(1);
area(rank, err_curve, 'LineStyle', 'none', 'FaceColor', ones(1,3)/3, ...
  'FaceAlpha', 0.7); hold on;
plot(rank, err_curve, 'k-', 'LineWidth', 1, 'Color', ones(1,3)/3);
plot(rank, err_baseline, 'k-', 'LineWidth', 2);
plot(rank, err_oracle, 'r--', 'LineWidth', 2);
hold off;
grid on;
axis square
axis([0, 1, 0, mean(err)])
xlabel('Normalized Image Rank');
ylabel('Angular Error');
drawnow;
ResizePlot(0.2, 3)
set(gca, 'XTick', 0.2:0.2:0.8)
set(gca, 'FontName', 'Times New Roman')
system('g4 edit ../docs/figures/confidence_err_gehler.png');
PrintCropped(['../docs/figures/confidence_err_gehler.png'], 300)

err = [];
conf = [];
base_folder = '/usr/local/google/home/barron/tmp/awb_vis/';
cameras = {'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200', ...
           'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'};
for i_camera = 1:length(cameras)
  folder = fullfile(base_folder, ['Cheng', cameras{i_camera}]);
  n = length(dir(fullfile(folder, '*error.txt')));
  for i = 1:n
    err(end+1) = ...
      load(fullfile(folder, [num2str(i, '%08d'), '_error.txt']));
    conf(end+1) = ...
      load(fullfile(folder, [num2str(i, '%08d'), '_confidence.txt']));
  end
end
err = err(:);
conf = conf(:);
entropy = -6.9315 - log(conf);
rank = ([1:length(err)] - 0.5) / length(err);

[~, sortidx] = sort(entropy, 1, 'ascend');
err_curve = cumsum(err(sortidx)) / length(err);
err_baseline = ([1:length(err)] - 0.5) / length(err) * mean(err);
[~, oracleidx] = sort(err, 1, 'ascend');
err_oracle = cumsum(err(oracleidx)) / length(err);

assert(abs(mean(err_baseline)*2 - mean(err)) < 1e-10)
assert(abs(err_curve(end) - mean(err)) < 1e-10);

fprintf('Cheng:  err = %0.3f / %0.3f\n', ...
  2*mean(err_curve), 2*mean(err_baseline))

figure(1);
area(rank, err_curve, 'LineStyle', 'none', 'FaceColor', ones(1,3)/3, ...
  'FaceAlpha', 0.7); hold on;
plot(rank, err_curve, 'k-', 'LineWidth', 1, 'Color', ones(1,3)/3);
plot(rank, err_baseline, 'k-', 'LineWidth', 2);
plot(rank, err_oracle, 'r--', 'LineWidth', 2);
hold off;
grid on;
axis square
axis([0, 1, 0, mean(err)])
xlabel('Normalized Image Rank');
ylabel('Angular Error');
drawnow;
ResizePlot(0.2, 3)
set(gca, 'XTick', 0.2:0.2:0.8)
set(gca, 'FontName', 'Times New Roman')
system('g4 edit ../docs/figures/confidence_err_cheng.png');
PrintCropped(['../docs/figures/confidence_err_cheng.png'], 300)
