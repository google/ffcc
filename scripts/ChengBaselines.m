function results = ChengResults
% Results taken from
% "Effective Learning-Based Illuminant Estimation Using Simple Features",
% Cheng et al 2015, CVPR.
% Table 2, from
% http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Cheng_Effective_Learning-Based_Illuminant_2015_CVPR_paper.pdf

results.methods = {'GW', 'WP', 'SoG', 'GGW', 'GE1', 'GE2', 'BD', 'LSR', 'PG', 'EG', 'BF', 'SS', 'NIS', 'CM', 'CD', 'Cheng2015'};
results.method_names.GW = 'Grey-world \cite{Buchsbaum80}';
results.method_names.WP = 'White-Patch \cite{Brainard86}';
results.method_names.SoG = 'Shades-of-Gray \cite{FinlaysonT04}';
results.method_names.GGW = 'General Gray-World \cite{Barnard02}';
results.method_names.GE1 = '1st-order Gray-Edge \cite{vandeWeijerTIP2007}';
results.method_names.GE2 = '2nd-order Gray-Edge \cite{vandeWeijerTIP2007}';
results.method_names.BD = 'Bright-and-dark Colors PCA \cite{Cheng14}';
results.method_names.LSR = 'Local Surface Reflectance Statistics \cite{Gao2014}'; %  "Efficient color constancy with local surface reflectance statistics"
results.method_names.PG = 'Pixels-based Gamut \cite{Gijsenij2010}'; %  Generalized gamut mapping using image derivative structures for color;
results.method_names.EG = 'Edge-based Gamut \cite{Gijsenij2010}'; %  Generalized gamut mapping using image derivative structures for color;
results.method_names.BF = 'Bayesian \cite{Gehler08}';
results.method_names.SS = 'Spatio-spectral Statistics \cite{Chakrabarti11}';
results.method_names.NIS = 'Natural Image Statistics \cite{GijsenijTPAMI2011}';
results.method_names.CM = 'Corrected-Moment \cite{Finlayson2013}';
results.method_names.CD = 'Color Dog \cite{BanicL15}'; % N. Banic and S. Loncaric. Color dog: Guiding the global illumination estimation to better accuracy. In International Conference on Computer Vision Theory
results.method_names.Cheng2015 = 'Cheng 2015 \cite{ChengCVPR2015}'; % Effective Learning-Based Illuminant Estimation Using Simple Features

results.Canon1DsMkIII.mean = [ 5.16 7.99 3.81 3.16 3.45 3.47 2.93 3.43 6.13 6.07 3.58 3.21 4.18 2.94 3.13 2.26];
results.Canon600D.mean = [ 3.89 10.96 3.23 3.24 3.22 3.21 2.81 3.59 14.51 15.36 3.29 2.67 3.43 2.76 2.83 2.43];
results.FujifilmXM1.mean = [  4.16 10.20 3.56 3.42 3.13 3.12 3.15 3.31 8.59 7.76 3.98 2.99 4.05 3.23 3.36 2.45];
results.NikonD5200.mean = [  4.38 11.64 3.45 3.26 3.37 3.47 2.90 3.68 10.14 13.00 3.97 3.15 4.10 3.46 3.19 2.51];
results.OlympusEPL6.mean = [  3.44 9.78 3.16 3.08 3.02 2.84 2.76 3.22 6.52 13.20 3.75 2.86 3.22 2.95 2.57 2.15];
results.PanasonicGX1.mean = [  3.82 13.41 3.22 3.12 2.99 2.99 2.96 3.36 6.00 5.78 3.41 2.85 3.70 3.10 2.84 2.36];
results.SamsungNX2000.mean = [  3.90 11.97 3.17 3.22 3.09 3.18 2.91 3.84 7.74 8.06 3.98 2.94 3.66 2.74 2.92 2.53];
results.SonyA57.mean = [  4.59 9.91 3.67 3.20 3.35 3.36 2.93 3.45 5.27 4.40 3.50 3.06 3.45 2.95 2.83 2.18];

results.Canon1DsMkIII.median = [ 4.15 6.19 2.73 2.35 2.48 2.44 2.01 2.51 4.30 4.68 2.80 2.67 3.04 1.98 1.72 1.57];
results.Canon600D.median = [ 2.88 12.44 2.58 2.28 2.07 2.29 1.89 2.72 14.83 15.92 2.35 2.03 2.46 1.85 1.85 1.62];
results.FujifilmXM1.median = [ 3.30 10.59 2.81 2.60 1.99 2.00 2.15 2.48 8.87 8.02 3.20 2.45 2.96 2.11 1.81 1.58];
results.NikonD5200.median = [ 3.39 11.67 2.56 2.31 2.22 2.19 2.08 2.83 10.32 12.24 3.10 2.26 2.40 2.04 1.94 1.65];
results.OlympusEPL6.median = [ 2.58 9.50 2.42 2.18 2.11 2.18 1.87 2.49 4.39 8.55 2.81 2.24 2.17 1.84 1.46 1.41];
results.PanasonicGX1.median = [ 3.06 18.00 2.30 2.23 2.16 2.04 2.02 2.48 4.74 4.85 2.41 2.22 2.28 1.77 1.69 1.61];
results.SamsungNX2000.median = [ 3.00 12.99 2.33 2.57 2.23 2.32 2.03 2.90 7.91 6.12 3.00 2.29 2.77 1.85 1.89 1.78];
results.SonyA57.median = [ 3.46 7.44 2.94 2.56 2.58 2.70 2.33 2.51 4.26 3.30 2.36 2.58 2.88 2.05 1.77 1.48];

results.Canon1DsMkIII.tri = [ 4.46 6.98 3.06 2.50 2.74 2.70 2.22 2.81 4.81 4.87 2.97 2.79 3.30 2.19 2.08 1.69];
results.Canon600D.tri = [ 3.07 11.40 2.63 2.41 2.36 2.37 2.12 2.95 14.78 15.73 2.40 2.18 2.72 2.12 2.07 1.80];
results.FujifilmXM1.tri = [ 3.40 10.25 2.93 2.72 2.26 2.27 2.41 2.65 8.64 7.70 3.33 2.55 3.06 2.33 2.20 1.81];
results.NikonD5200.tri = [ 3.59 11.53 2.74 2.49 2.52 2.58 2.19 3.03 10.25 11.75 3.36 2.49 2.77 2.30 2.14 1.82];
results.OlympusEPL6.tri = [ 2.73 9.54 2.59 2.35 2.26 2.20 2.05 2.59 4.79 10.88 3.00 2.28 2.42 1.92 1.72 1.55];
results.PanasonicGX1.tri = [ 3.15 14.98 2.48 2.45 2.25 2.26 2.31 2.78 4.98 5.09 2.58 2.37 2.67 2.00 1.87 1.71];
results.SamsungNX2000.tri = [ 3.15 12.45 2.45 2.66 2.32 2.41 2.22 3.24 7.70 6.56 3.27 2.44 2.94 2.10 2.05 1.87];
results.SonyA57.tri = [ 3.81 8.78 3.03 2.68 2.76 2.80 2.42 2.70 4.45 3.45 2.57 2.74 2.95 2.16 2.03 1.64];

results.Canon1DsMkIII.b25 = [ 0.95 1.56 0.66 0.64 0.81 0.86 0.59 1.06 1.05 1.38 0.76 0.88 0.78 0.65 0.59 0.54];
results.Canon600D.b25 = [ 0.83 2.03 0.64 0.63 0.73 0.80 0.55 1.17 9.98 11.23 0.69 0.68 0.78 0.65 0.54 0.48];
results.FujifilmXM1.b25 = [ 0.91 1.82 0.87 0.73 0.72 0.70 0.65 0.99 3.44 2.30 0.93 0.81 0.86 0.75 0.56 0.53];
results.NikonD5200.b25 = [ 0.92 1.77 0.72 0.63 0.79 0.73 0.56 1.16 4.35 3.92 0.92 0.86 0.74 0.66 0.58 0.52];
results.OlympusEPL6.b25 = [ 0.85 1.65 0.76 0.72 0.65 0.71 0.55 1.15 1.42 1.55 0.91 0.78 0.76 0.51 0.49 0.43];
results.PanasonicGX1.b25 = [ 0.82 2.25 0.78 0.70 0.56 0.61 0.67 0.82 2.06 1.76 0.68 0.82 0.79 0.64 0.51 0.47];
results.SamsungNX2000.b25 = [ 0.81 2.59 0.78 0.77 0.71 0.74 0.66 1.26 2.65 3.00 0.93 0.75 0.75 0.66 0.55 0.51];
results.SonyA57.b25 = [ 1.16 1.44 0.98 0.85 0.79 0.89 0.78 0.98 1.28 0.99 0.78 0.87 0.83 0.59 0.48 0.46];

results.Canon1DsMkIII.w25 = [ 11.00 16.75 8.52 7.08 7.69 7.76 6.82 7.30 14.16 13.35 7.95 6.43 9.51 6.93 7.94 5.17];
results.Canon600D.w25 = [ 8.53 18.75 7.06 7.58 7.48 7.41 6.50 7.40 18.45 18.66 7.93 5.77 5.76 6.28 7.06 5.63];
results.FujifilmXM1.w25 = [ 9.04 18.26 7.55 7.62 7.32 7.23 7.30 7.06 13.40 13.44 8.82 5.99 9.37 7.66 8.24 5.73];
results.NikonD5200.w25 = [ 9.69 21.89 7.69 7.53 8.42 8.21 6.73 7.57 15.93 24.33 8.18 6.90 10.01 8.64 7.80 5.98];
results.OlympusEPL6.w25 = [ 7.41 18.58 6.78 6.69 6.88 6.47 6.31 6.55 15.42 30.21 8.19 6.14 7.46 7.39 6.43 5.15];
results.PanasonicGX1.w25 = [ 8.45 20.40 7.12 6.86 7.03 6.86 6.66 7.42 12.19 11.38 8.00 5.90 8.74 7.81 6.98 5.65];
results.SamsungNX2000.w25 = [ 8.51 20.23 6.92 6.85 7.00 7.23 6.48 7.98 13.01 16.27 8.62 6.22 8.16 6.27 6.95 5.96];
results.SonyA57.w25 = [ 9.85 21.27 7.75 6.68 7.18 7.14 6.13 7.32 11.16 9.83 8.02 6.17 7.18 6.89 7.04 5.03];

results.methods{end+1} = 'Shi2016';
results.method_names.Shi2016 = 'Shi \etal\, 2016 \cite{ShiECCV2016}';

cameras = fieldnames(results)';
cameras = cameras(3:end);
for camera = cameras
  camera = camera{1};
  results.(camera).mean(end+1) = 2.24;
  results.(camera).median(end+1) = 1.46;
  results.(camera).tri(end+1) = 1.68;
  results.(camera).b25(end+1) = 0.48;
  results.(camera).w25(end+1) = 6.08;
end
