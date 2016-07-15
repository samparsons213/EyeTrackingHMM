function [Sigma] = thresholdCovMatrix(Sigma, min_eig)
% Thresholds the eigenvalues of the matrix Sigma to be greater than or
% equal to min_eig. Sigma is assumed to be square, symmetric, and almost
% pos. def. - i.e. any non-positive eigenvalues are very close to being
% positive. This assumption does not affect the running of the function,
% though results may be undesired if they do not hold.

% Inputs:

% Sigma:        n by n matrix assumed to be symmetric and almost pos. def.

% min_eig:      positive real number giving the smallest acceptable
%               eigenvalue of Sigma

% Outputs:

% Sigma:        n by n symmetric pos. def. matrix

% Author:       Sam Parsons
% Date created: 12/07/16
% Last amended: 12/07/16

    [V, D] = eig(Sigma);
    D = diag(max(diag(D), min_eig));
    Sigma = V * D * V';

end