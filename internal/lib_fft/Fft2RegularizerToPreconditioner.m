function V_pre = Fft2RegularizerToPreconditioner(F_reg)
% Given a regularizer in the frequency domain (a per-FFT-bin non-negative
% value indicating how much that frequency will be regularized) we construct a
% Jacobi preconditioner, which is the the inverse diagonal of that
% quadratic regularizer.
% "Preconditioning" is a bit of a misnomer here, as we are actually
% "pretransforming" our model, and performing LBFGS in this transformed space.
% As a result, the "preconditioner" scalings that we produce are actually the
% square-root of what the actual preconditioner should be, as these scalings are
% effectively applied twice.

assert(all(F_reg(:) > 0));  % Prevent divide by zero.

% Because of the sqrt(2) scaling in Fft2ToVec
V_pre = sqrt(sqrt(2) ./ Fft2ToVec(complex(F_reg, F_reg)));

% We need to special-case the 4 FFT elements that are present only once.
n = size(F_reg,1);
scale_idx = [1; n/2 + 1; n/2 + 2; n + 2];
V_pre(scale_idx,:) = V_pre(scale_idx,:) ./ sqrt(sqrt(2));
