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
