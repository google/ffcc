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

function str = BlobToString(blob, print_log, indent)
% Serializes a struct of (structs of) numers and strings into an easy to read
% string. if print_log==true then values are printed in the form of 2^{log2(x))
% instead of x. The output string has the property that, if evaluated, the
% string returns the blob (eval(BlobToString(blob)) == blob) ignoring roundoff
% error during printing. The "indent" argument is a string of spaces which
% is used to properly indent nested elements of the blob.
%
% Example useage:
%   fprintf('%s\n', BlobToString(blob));

if ~exist('print_log', 'var')
  print_log = false;
end

if ~exist('indent', 'var')
  indent = '';
end

tab = '  ';  % The indentation used when formatting.
str = '';

if ischar(blob)
  str = [str, '''', blob, ''''];
elseif isnumeric(blob) || islogical(blob)
  if numel(blob) == 1
    val = blob(1,1);
    if print_log && islogical(blob)
      str = [str, sprintf('%d', val)];
    elseif print_log && (val ~= 0)
      str = [str, sprintf('2^%g', log2(val))];
    else
      str = [str, sprintf('%g', val)];
    end
  elseif (ndims(blob) == 4)
    str = [str, sprintf('cat(4, ...\n'), indent];
    for k = 1:size(blob, 4)
      str = [str, BlobToString(blob(:,:,:,k), print_log, [indent, tab])];
      if k < size(blob, 4)
        str = [str, sprintf(', ...\n'), indent];
      else
        str = [str, sprintf(')')];
      end
    end
  elseif (ndims(blob) == 3)
    str = [str, sprintf('cat(3, ...\n'), indent];
    for k = 1:size(blob, 3)
      str = [str, BlobToString(blob(:,:,k), print_log, [indent, tab])];
      if k < size(blob, 3)
        str = [str, sprintf(', ...\n'), indent];
      else
        str = [str, ')'];
      end
    end
  elseif (ndims(blob) <= 2)
    str = [str, '['];
    for i = 1:size(blob, 1)
      for j = 1:size(blob, 2)
        val = blob(i,j);
        str = [str, BlobToString(val, print_log, [indent, tab])];
        if j < size(blob, 2)
          str = [str, ', '];
        end
      end
      if i < size(blob, 1)
        str = [str, sprintf('; ...\n'), indent];
      end
    end
    str = [str, ']'];
  else
    assert(false)
  end
elseif iscell(blob)
  assert(ndims(blob) <= 2);
  str = [str, '{'];
  for i = 1:size(blob, 1)
    for j = 1:size(blob, 2)
      str = [str, BlobToString(blob{i,j}, print_log, [indent, tab])];
      if j < size(blob, 2)
        if size(blob{i,j},1) == 1
          str = [str, sprintf(', ')];
        else
          str = [str, sprintf(', ...\n'), indent];
        end
      end
    end
    if i < size(blob, 1)
      str = [str, sprintf('; ...\n'), indent];
    end
  end
  str = [str, sprintf('}')];
else
  str = [str, sprintf('struct( ...\n'), indent];
  fields = fieldnames(blob);
  for i_field = 1:length(fields)
    field = fields{i_field};
    str = [str, sprintf('''%s'', ', field)];
    % Initializing a field of a struct to a cell causes matlab to behave
    % bizarrely and construct an array of structs. Apparently this can be
    % avoided by wrapping cell fields in an extra set of curly brackets.
    if ~((isnumeric(blob.(field)) || islogical(blob.(field))) ...
        && (size(blob.(field), 1) == 1))
      str = [str, sprintf('...\n'), indent, tab];
    end
    if iscell(blob.(field))
      str = [str, '{'];
    end
    str = [str, BlobToString(blob.(field), print_log, [indent, tab, tab])];
    if iscell(blob.(field))
      str = [str, '}'];
    end
    if i_field < length(fields)
      str = [str, sprintf(', ...\n'), indent];
    end
  end
  str = [str, sprintf(' ...\n)'), indent];
end
