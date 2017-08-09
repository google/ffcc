function results = GehlerShiBaselines
% Results taken from
% "Effective Learning-Based Illuminant Estimation Using Simple Features",
% Cheng et al 2015, CVPR.
% Table 1, from
% http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Cheng_Effective_Learning-Based_Illuminant_2015_CVPR_paper.pdf

results.method_names.GW = 'Grey-world \cite{Buchsbaum80}';
results.method_names.WP = 'White-Patch \cite{Brainard86}';
results.method_names.SoG = 'Shades-of-Gray \cite{FinlaysonT04}';
results.method_names.GGW = 'General Gray-World \cite{Barnard02}';
results.method_names.GE1 = '1st-order Gray-Edge \cite{vandeWeijerTIP2007}';
results.method_names.GE2 = '2nd-order Gray-Edge \cite{vandeWeijerTIP2007}';
results.method_names.BD = 'Cheng \etal\, 2014 \cite{Cheng14}';
results.method_names.LSR = 'LSRS \cite{Gao2014}'; %'Local Surface Reflectance Statistics'; %  "Efficient color constancy with local surface reflectance statistics"
results.method_names.PG = 'Pixels-based Gamut \cite{Gijsenij2010}'; %  Generalized gamut mapping using image derivative structures for color;
results.method_names.EG = 'Edge-based Gamut \cite{Gijsenij2010}'; %  Generalized gamut mapping using image derivative structures for color;
results.method_names.IG = 'Interesection-based Gamut \cite{Gijsenij2010}'; %  Generalized gamut mapping using image derivative structures for color;
results.method_names.SVR = 'Support Vector Regression \cite{FuntX04}'; %  Estimating illumination chromaticity via support vector regression
results.method_names.BF = 'Bayesian \cite{Gehler08}';
results.method_names.CART = 'CART-based Combination \cite{Bianco2010}'; % Automatic color constancy algorithm selection and combination.
results.method_names.BOT = 'Bottom-up+Top-down \cite{VSV2007a}'; % [40] J. Van De Weijer, C. Schmid, and J. Verbeek. Using highlevel visual information for color constancy
results.method_names.SS = 'Spatio-spectral Statistics \cite{Chakrabarti11}';
results.method_names.EX = 'Exemplar-based \cite{Joze2014}'; % Exemplar-based colour constancy and multiple illumination.
results.method_names.NIS = 'Natural Image Statistics \cite{GijsenijTPAMI2011}';
results.method_names.CM = 'Corrected-Moment \cite{Finlayson2013}';
results.method_names.CD = 'Color Dog  \cite{BanicL15}'; % N. Banic and S. Loncaric. Color dog: Guiding the global illumination estimation to better accuracy. In International Conference on Computer Vision Theory
results.method_names.Chakrabarti2015 = 'Chakrabarti \etal\, 2015 \cite{Chakrabarti2015}';
results.method_names.Cheng2015 = 'Cheng \etal\, 2015 \cite{ChengCVPR2015}'; % Effective Learning-Based Illuminant Estimation Using Simple Features
results.method_names.Barron2015 = 'CCC \cite{BarronICCV2015}';
results.method_names.Shi2016 = 'Shi \etal\, 2016 \cite{ShiECCV2016}';
results.method_names.Bianco2015 = 'Bianco \etal\, 2015 \cite{Bianco2015}';
results.method_names.Yang2015 = 'Yang \etal\, 2015 \cite{Yang2015}';

results.testtimes.GW = 0.15;
results.testtimes.WP = 0.16;
results.testtimes.SoG = 0.47;
results.testtimes.GGW = 0.91;
results.testtimes.GE1 = 1.05;
results.testtimes.GE2 = 1.26;
results.testtimes.BD =  0.24;
results.testtimes.LSR = 0.22;
results.testtimes.LSR = 2.65;
results.testtimes.EG = 3.64;
results.testtimes.BF = 96.57;
results.testtimes.SS = 6.88;
results.testtimes.NIS = 1.49;
results.testtimes.CM = 0.77;
results.testtimes.Cheng2015 = 0.25;
results.testtimes.Chakrabarti2015 = 0.3;
results.testtimes.Barron2015 = 0.5207;
results.testtimes.Shi2016 = 3; % "Our unoptimized C++ code takes approximately 3 secs to process an image on a GPU."
results.testtimes.Yang2015 = 0.88;
% featurize times (min, med)
%  0.4869    0.5751
% filter times (min, med)
%  0.0338    0.0743

results.traintimes.LSR = 1345;
results.traintimes.EG = 1986;
results.traintimes.BF = 764;
results.traintimes.SS = 3159;
results.traintimes.NIS = 10749;
results.traintimes.CM = 584;
results.traintimes.Cheng2015 = 245;
results.traintimes.Barron2015 = 2168;

results.metrics.GW  = [ 6.36 6.28 6.28 2.33 10.58];
results.metrics.WP  = [ 7.55 5.68 6.35 1.45 16.12];
results.metrics.SoG = [ 4.93 4.01 4.23 1.14 10.20];
results.metrics.GGW = [ 4.66 3.48 3.81 1.00 10.09];
results.metrics.GE1 = [ 5.33 4.52 4.73 1.86 10.03];
results.metrics.GE2 = [ 5.13 4.44 4.62 2.11 9.26];
results.metrics.BD  = [3.52 2.14 2.47 0.50 8.74];
results.metrics.LSR = [3.31 2.80 2.87 1.14 6.39];
results.metrics.PG  = [4.20 2.33 2.91 0.50 10.72];
results.metrics.EG  = [6.52 5.04 5.43 1.90 13.58];
results.metrics.IG  = [4.20 2.39 2.93 0.51 10.70];
results.metrics.SVR = [8.08 6.73 7.19 3.35 14.89];
results.metrics.BF  = [4.82 3.46 3.88 1.26 10.49];
results.metrics.SS  = [3.59 2.96 3.10 0.95 7.61];
results.metrics.CART = [3.90 2.91 3.21 1.02 8.27];
results.metrics.NIS = [4.19 3.13 3.45 1.00 9.22];
results.metrics.BOT = [3.48 2.47 2.61 0.84 8.01];
results.metrics.EX  = [2.89 2.27 2.42 0.82 5.97];
results.metrics.CM  = [2.86 2.04 2.22 0.70 6.34];
results.metrics.Chakrabarti2015 = [2.56 1.67 1.89 0.52 6.07];
results.metrics.Cheng2015 = [2.42 1.65 1.75 0.38 5.87];
results.metrics.Barron2015 = [1.95 1.22 1.38 0.35 4.76];
results.metrics.Shi2016 = [1.90 1.12 1.33 0.31 4.84];
results.metrics.Bianco2015 = [2.63 1.98 nan nan nan];
results.metrics.Yang2015 = [4.6 3.1  nan nan nan];
