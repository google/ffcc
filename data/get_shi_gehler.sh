# This code release includes a preprocessed version of the Gehler-Shi dataset.
# For full transparency, here we detail how to reproduce this data. This guide
# can be run as a shell script, but there's some commented out instructions and
# Matlab at the end that you'll need to handle manually.

# This script depends on the data found at *both* of these addresses:
# http://www.cs.sfu.ca/~colour/data/shi_gehler/
# http://files.is.tue.mpg.de/pgehler/projects/color/index.html
# If these links go down, please let them (and me) know.

# Run this script from inside /ffcc/data/

# Yes, the name of this folder matters.
mkdir shi_gehler
cd shi_gehler

# Downloading will take some time, go grab a coffee.
wget \
  http://www.cs.sfu.ca/~colour/data/shi_gehler/groundtruth_568.zip \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon1d.zip \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_1.zip \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_2.zip \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_3.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_1.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_2.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_3.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_4.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_5.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_6.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_7.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_RAW_8.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/ColorCheckerDatabase_MaskCoordinates.zip \
  ftp://ftp.tuebingen.mpg.de/kyb/pgehler/ColorCheckerDatabase/code.zip

# Unzip and reformat all of the data.
unzip '*.zip'
rm *.zip
mv real_illum_568..mat real_illum_568.mat
mkdir ./images
mv cs/chroma/data/canon_dataset/568_dataset/png/*.png ./images/
mv experiment/threefoldCVsplit.mat .
rm -rf ./cs
rm -rf bayesiancc misc greyworld experiment
rm -rf __MACOSX  # "Sent from my iPhone."

# Now check that paths.gehler_shi in /internal/DataPaths.m is set to the
# global path that you unpacked all of the raw data into.

# The downloaded data needs to be preprocessed before any experiments can be
# run. To do this, run
# scripts/GehlerShiImportData.m
# and if you want to run the "Deep" model that uses metadata, also run
# scripts/GehlerShiComputeEXIF.m

# After all of this is done, you can safely delete all but
#/shi_gehler/preprocessed/* and /shi_gehler/threefoldCVsplit.mat
