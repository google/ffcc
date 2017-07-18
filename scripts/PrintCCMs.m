clearvars;

FOLDERS = {'../projects/GehlerShi/tags/', '../projects/Cheng/tags/'};

cameras.Canon1D = 'GehlerShi, Canon1D';
cameras.Canon5D = 'GehlerShi, Canon5D';
cameras.Canon600D = 'Cheng, Canon600D';
cameras.OlympusEPL6 = 'Cheng, OlympusEPL6';
cameras.SamsungNX2000 = 'Cheng, SamsungNX2000';
cameras.Canon1DsMkIII = 'Cheng, Canon1DsMkIII';
cameras.FujifilmXM1 = 'Cheng, FujifilmXM1';
cameras.NikonD5200 = 'Cheng, NikonD5200';
cameras.PanasonicGX1 = 'Cheng, PanasonicGX1';
cameras.SonyA57 = 'Cheng, SonyA57';

output_filename = '../docs/ccms.tex';

system(['g4 edit ', output_filename]);
fid = fopen(output_filename, 'w');

fprintf(fid, '\\begin{align}\n');

first = true;
for i_folder = 1:length(FOLDERS)
  folder = FOLDERS{i_folder};
  dirents = dir(fullfile(folder, '*CCM.txt'));
  filenames = {dirents.name};
  for i_filename = 1:length(filenames)
    filename = filenames{i_filename};
    CCM = load(fullfile(folder, filename));
    camera = filename(1:find(filename == '_', 1, 'first')-1);
    if isfield(cameras, camera)
      if ~first
        fprintf(fid, ' \\\\ \n');
      else
        first = false;
      end
      fprintf(fid, '\\small{\\mathrm{%s}} \n', cameras.(camera));
      fprintf(fid, '\\begin{bmatrix}\n');
      for i = 1:3
        for j = 1:3
          if CCM(i,j) >= 0
            fprintf(fid, '\\phantom{-}');
          end
          fprintf(fid, '%0.4f', CCM(i,j));
          if j < 3
            fprintf(fid, ' & ');
          end
        end
        fprintf(fid, ' \\\\ \n');
      end
      fprintf(fid, '\\end{bmatrix} \\nonumber \n');
    end
  end
end
fprintf(fid, '\\end{align}\n');

fclose(fid);
