function gehlershi_sensor = GehlerShiSensor

% The blacks levels for the two cameras used by the Gehler-Shi dataset.
gehlershi_sensor.BLACK_LEVEL.Canon1D = 0;
gehlershi_sensor.BLACK_LEVEL.Canon5D = 129;
% The saturation levels used by the two cameras in the dataset, which happen to
% be the same. Past papers use 3580 as the saturation value, which does not
% appear to be correct.
gehlershi_sensor.SATURATION.Canon1D = 3692;
gehlershi_sensor.SATURATION.Canon5D = 3692;
% Our estimate of the CCMs of the two cameras used in the dataset. This is not
% used for training models for this project, but is useful for preprocessing.
working_dir = pwd;
gehlershi_sensor.CCMs.Canon1D = ...
  load(fullfile(working_dir(1:strfind(working_dir, 'matlab')+5), ...
  'projects', 'GehlerShi', 'tags', 'Canon1D_CCM.txt'));
gehlershi_sensor.CCMs.Canon5D = ...
  load(fullfile(working_dir(1:strfind(working_dir, 'matlab')+5), ...
  'projects', 'GehlerShi', 'tags', 'Canon5D_CCM.txt'));
