function TestAll

% This script must be run from within /internal/.
path = pwd;
assert(~isempty(strfind(path, 'awb/tool/matlab/internal')));

addpath('lib_fft');
addpath('lib_flatten');

dirents = [dir('*_Test.m'); ...
           dir('lib_fft/*_Test.m'); ...
           dir('lib_flatten/*_Test.m')];

filenames = {dirents.name};
filenames = cellfun(@(x)x(1:end-2), filenames, 'UniformOutput', false);

failed_tests = {};
for i = 1:length(filenames)
  fprintf('%s: ', filenames{i});
  try
    eval(filenames{i});
  catch me
    fprintf('Test Failed\n');
    failed_tests{end+1} = filenames{i};
  end
end

if isempty(failed_tests)
  fprintf('All Tests Passed\n');
else
  fprintf('Some tests failed:\n');
  for i = 1:length(failed_tests)
    fprintf('  %s\n', failed_tests{i})
  end
end
