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

function filenames = ScrapeFolder(folder, expression)
% Scans through the input folder and all of its sub-folders for filenames
% matching the input expression. For example:
%   filenames = ScrapeFolder('/tmp/', '*.png');
% returns a cell array of full paths to all png files in /tmp/ and in
% any subfolder of /tmp/. This function can also be called with a cell array
% of expressions, for example:
%   filenames = ScrapeFolder('/tmp/',{'*.png', '*.jpg'});
% returns a cell array of full paths to all png or jpg files in /tmp/ and in
% any subfolder of /tmp/.

if iscell(folder)

  filenames = {};
  for i_folder = 1:length(folder)
    filenames{i_folder} = ScrapeFolder(folder{i_folder}, expression);
  end
  filenames = cat(2, filenames{:});

else

  if isempty(expression)
    % This check catches the base case in the following recursion.
    filenames = {};
  elseif iscell(expression)
    % If the expression is a cell array, recursively call this function on the
    % first element and the rest of the elements of the cell array, and then
    % concatenate the two results.
    filenames_first = ScrapeFolder(folder, expression{1});
    filenames_rest = ScrapeFolder(folder, expression(2:end));
    filenames = cat(2, filenames_first(:)', filenames_rest(:)');
  else
    % Produce a large string containing the full path of "folder" and of all
    % its sub-folders, separated by ":".
    subfolder_string = genpath(folder);
    filenames = {};
    while true
      [subfolder, subfolder_string] = strtok(subfolder_string, ':');
      if isempty(subfolder)
        break
      end
      dirents = dir(fullfile(subfolder, expression));
      filenames{end+1} = cellfun(...
        @(x) fullfile(subfolder, x), {dirents.name}, 'UniformOutput', false);
    end
    filenames = cat(2, filenames{:});
  end

  % Because different operating systems might sort files and directories
  % differently, and because we may want to apply temporal filtering to a
  % sequence of images, we sort the filenames before returning them.
  filenames = sort(filenames);

end
