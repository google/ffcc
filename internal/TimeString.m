function str = TimeString()
% Produces a string encoding the current time. This is useful for coming up with
% tags, filenames, or folders for experiments.

str = datestr(now, 'yyyy_mm_dd_HH_MM_SS_FFF');

% Prevents TimeString() from being called within a millisecond of itself.
pause(2/1000)
