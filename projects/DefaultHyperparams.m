function params = DefaultHyperparams(params)
% Default hyperparameters for a new project. This file mostly exists to
% provide documentation to each hyperparameter, as these default values may
% be very far from optimal. Initializing to the hyperparameters in a
% pre-existing project with similar data may be preferable.

% The multiplier for all losses computed with the convex cross-entropy loss
% function (cross-entropy) relative to the non-convex loss function.
params.HYPERPARAMS.CROSSENT_MULTIPLIER = 2^0;

% The multiplier for all losses computed with the non-convex Von Mises negative
% log-likelihood loss. Setting this == CROSSENT_LOSS_MULTIPLIER is reasonable
% as both are in "nats" (https://en.wikipedia.org/wiki/Nat_(unit)).
params.HYPERPARAMS.VONMISES_MULTIPLIER = 2^0;

% The multipliers on the total varaition regularizer of the filter. Larger
% values mean a more smooth response to the scene content. The first parameter
% corresponds to pixel intensity, and the second corresponds to edge
% intensity.
params.HYPERPARAMS.FILTER_MULTIPLIERS = ...
  repmat(params.HISTOGRAM.NUM_BINS^-4, 1, 2);

% The multiplier on the total variation regularizer of the black body bias.
% Larger values encourage the black body bias to be smoother which will reduce
% the model's bias towards particular white points.
params.HYPERPARAMS.BIAS_MULTIPLIER = 2^0;

% These hyperparamteres correspond to simpler L2 regularizers of the filters,
% bias, and log-gain maps. These usually have a small impact on
% performance, but can help, and must be set to positive values to prevent
% divide-by-zero errors during preconditioning.
params.HYPERPARAMS.FILTER_SHIFTS = [ 2^-8, 2^-8];
params.HYPERPARAMS.BIAS_SHIFT = 2^-8;

% The amount that is added to (or clamped by, depending on the project) the
% diagonal of the von Mises distributions during training.
params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS = 1;
