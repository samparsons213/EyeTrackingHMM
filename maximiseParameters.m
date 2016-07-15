function [pi1, P, means, vars] = maximiseParameters(uni, pwise, y_hmm,...
    t, min_eig)
% Maximises the data likelihood wrt the parameters, as per EM

% Inputs:

% uni:          T by dim_x array of univariate latent marginals 
%               P(X_t | Y_{1:T})

% pwise:        dim_x by dim_x by (T-1) array of pairwise latent marginals
%               P(X_t, X_t+1 | Y_{1:T})

% y_hmm:        n_obs by 2 by (n_dirs+1) array of transformed data,
%               with the first element of the third dimension
%               corresponding to the no-movement latent state, and
%               the following elements corresponding to the
%               elements of l_dirs as per the input argument of
%               yTransformHMM.m

% t:            n_obs by 1 integer array giving the time passed
%               corresponding to each difference vector

% min_eig:      positive real number giving the minimum acceptable
%               eigenvalue for esitmated covariance matrices

% Outputs:

% pi1:          1 by dim_x probability vector for latent prior at t=1

% P:            dim_x by dim_x transition matrix

% means:        (n_dirs+1) by 2 array of mean vectors, one for each
%               latent state

% vars:         2 by 2 by (n_dirs+1) array of covariance matrices,
%               one for each latent state

% Author:       Sam Parsons
% Date created: 12/07/16
% Last amended: 12/07/16

    pi1 = uni(1, :);
    P = sum(pwise, 3);
    dim_x = length(pi1);
    P = P ./ repmat(sum(P, 2), 1, dim_x);
    % to find mle mean and variance parameters is very similar to standard
    % mixture of Gaussians.
    [n_obs, dim_y, dim_x] = size(y_hmm);
    weights = uni ./ repmat(sum(uni, 1), n_obs, 1);
    t = repmat(t ./ 4, 1, dim_y);
    root_t = sqrt(t);
    means = zeros(dim_x, dim_y);
    vars = zeros(dim_y, dim_y, dim_x);
    for x_dim = 1:dim_x
         means(x_dim, :) = weights(:, x_dim)' * (y_hmm(:, :, x_dim) ./ t);
         centred_y_i = y_hmm(:, :, x_dim) ./ root_t;
         centred_y_i = centred_y_i - root_t .* repmat(means(x_dim, :), n_obs, 1);
         vars(:, :, x_dim) = centred_y_i' *...
             (repmat(weights(:, x_dim), 1, dim_y) .* centred_y_i);
         vars(:, :, x_dim) = thresholdCovMatrix(vars(:, :, x_dim), min_eig);
    end

end