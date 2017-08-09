# Fast Fourier Color Constancy Matlab Toolbox

The Fast Fourier Color Constancy (FFCC) Matlab Toolbox includes the following
functionalities:

*  Tune() - Cross-validation and parameter tuning.
*  Train() - Training.
*  Visualize() - Visualizing cross-validation or training/test performance.

This code depends on the "minFunc" library from Mark Schmidt:
https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
Either add this to the path manually or place it inside the root /ffcc/
directory.

## Training & Cross Validation

The following section discusses training and cross validation.

### Preparation and Folder Structure
The training folder structure should look similar to the following:


```
- train_folder/
            + folder_1/
            - folder_2/
                      | 000001.png
                      | 000001.txt
                      ...
```

The script will parse down the subfolders and look for *.png and *.txt files,
which corresponds to the linear thumbnail and the color of global illuminant.
This data has been provided for the Gehler-Shi dataset, alongside a script
for generating this data "from scratch" (see /ffcc/data/).

#### Linear Thumbnail

The PNG file is usually a small resolution thumbnail with linear data after
black level removed. This is very important since this assumption enables shift
invariance. This PNG file needs to match to the format of your imaging pipeline
spec.

#### Global Illuminant

The text file descibes the color of the global illuminant. It looks like this:

```
0.500249
0.819743
0.278876
```

The equation to convert between white balance gain and the color of illuminant is
as follows:

$$
L = z / |z|
$$

where \\(z = [1/R_{gain}, 1/G_{gain}, 1/B_{gain}]\\).

### Project Folder

To allow FFCC Toolbox to support wide range of projects, we separate out the
core algorithms from the project specific implementations. The project specific
implementations are placed under projects/ folder.

The following scripts all take as input the string of some project name, which
must correspond exactly to a folder in projects/, and which must prefix all
filenames in the projects/ subfolder.

### Tuning and cross-validation
Tune() performs coordinate descent over hyperparameters, and can also be used to
sanity-check the current hyperparameter set as a diagnostic. A side effect of
tuning/cross-validation is that it will produce a set of error metrics, of the
following form:

```
angular_err.mean = 2.0749
angular_err.median = 1.1279
angular_err.tri = 1.3221
angular_err.b25 = 0.2924
angular_err.w25 = 5.4809
angular_err.q95 = 7.3257
vonmises_nll.mean = -2.3466
vonmises_nll.median = -3.2563
vonmises_nll.tri = -3.2106
vonmises_nll.b25 = -4.6225
vonmises_nll.w25 = 1.6908
vonmises_nll.q95 = 1.8778
```

The first element of each printed label is a kind of error measure, and the
second element is a surface statistic on that error metric. Please see
CrossValidate.m and ErrorMetrics.m for detailed descriptions.
An angular_err less than 2 degrees is not perceptible, and vonmisses_nll is
is measured in negative-nats for an individual training point. If the error is
more than 2 degrees, you might consider:

* Allow Tune() to keep brute-force searching for better parameter settings.
  Better parameters need to be manually copy-pasted into the *Hyperparams.m
  file to be used in later training and tuning, as described in Tune(). Tuning
  may be slow, so consider running it overnight or for several days.
* Augment and diversify your dataset. Make sure your data is equally presented
  different lighting conditions, instead of heavily favoring one particular
  group.

You can read further tuning description in Tune.m or CrossValidate.m

### Training
Once Tune() is producing reasonable cross-validation results, use Train()
to train a final model which will be written to disk. See Train() for details.

### Visualization
Visualize() provides visualizing and testing functinality. Given a project name,
It will do all necessary training (or cross-validation) and produce a
visualization for each image. If the project uses cross-validation, then each
image is evaluated using the model that did not include that image in its
training fold. If the project uses training/test splits, then all images are
evaluated using the model trained on the training set.
If params.TRAINING.DUMP_EXHAUSTIVE_VISUALIZATION
is set to true, this script will dump an extensive set of images and files
describing the output of the model, which can be useful for debugging.
See CrossValidate() for details.
