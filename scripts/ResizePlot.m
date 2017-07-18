function ResizePlot(borderSize, PAPERBASESIZE)

set(gca, 'Position', [borderSize borderSize 1-2*borderSize 1-2*borderSize]);
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, PAPERBASESIZE, PAPERBASESIZE]);
