% A script for dumping the output predictions of a few models used in the paper.

clearvars;
addpath(genpath('../'));
addpath(genpath('../../../../../../third_party/matlab/minFunc'));

try
  Tune('GehlerShiThumb', false, ...
    ['params.TRAINING.DO_NOT_TUNE = true; ', ...
     'params.DEBUG.OUTPUT_SUMMARY = ''/tmp/GehlerShi_ModelQ.txt'';']);
catch me
  me;
end

try
  Tune('GehlerShi', false, ...
    ['params.TRAINING.DO_NOT_TUNE = true; ', ...
     'params.DEBUG.OUTPUT_SUMMARY = ''/tmp/GehlerShi_ModelJ.txt'';'])
catch me
  me;
end

try
  Tune('GehlerShiDeepE', false, ...
    ['params.TRAINING.DO_NOT_TUNE = true; ', ...
     'params.DEBUG.OUTPUT_SUMMARY = ''/tmp/GehlerShi_ModelO.txt'';'])
catch me
  me;
end

try
  Tune('GehlerShiDeepES', false, ...
    ['params.TRAINING.DO_NOT_TUNE = true; ', ...
     'params.DEBUG.OUTPUT_SUMMARY = ''/tmp/GehlerShi_ModelP.txt'';'])
catch me
  me;
end
