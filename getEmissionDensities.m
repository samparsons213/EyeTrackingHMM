function [emission_densities] = getEmissionDensities(y_hmm, t, means, vars,...
    p0)
% Returns the emission densities for all observed difference vectors
% conditional on the latent direction vector. After transformation by
% yTransformHMM.m the transformed observed difference vectors in y_hmm are
% modelled to follow Gaussians with means and vars given as arguments.

% Inputs:

% y_hmm:                n_obs by 2 by (n_dirs+1) array of transformed data,
%                       with the first element of the third dimension
%                       corresponding to the no-movement latent state, and
%                       the following elements corresponding to the
%                       elements of l_dirs as per the input argument of
%                       yTransformHMM.m

% t:                    n_obs by 1 integer array giving the time passed
%                       corresponding to each difference vector

% means:                (n_dirs+1) by 2 array of mean vectors, one for each
%                       latent state

% vars:                 2 by 2 by (n_dirs+1) array of covariance matrices,
%                       one for each latent state

% p0:                   n_obs by (n_dirs+1) logical array giving true when
%                       the probability of a latent direction given an
%                       observed difference vector is zero (i.e. the dot
%                       product between the vector and the latent direction
%                       is non-positive for movement states). First col is
%                       always false as all observations are logically
%                       consistent with target staying still

% Outputs:

% emission_densities:   n_obs by (n_dirs+1) array of emission densities for
%                       the observed difference vectors given each latent
%                       state

% Author:               Sam Parsons
% Date created:         03/07/16
% Last amended:         14/07/16

    [n_obs, ~, dim_x] = size(y_hmm);
    emission_densities = zeros(n_obs, dim_x);
    t = t ./ 4;
    for x_dim = 1:dim_x
        emission_densities(:, x_dim) =...
            mvnpdf(y_hmm(:, :, x_dim),...
            repmat(t, 1, 2) .* repmat(means(x_dim, :), n_obs, 1),...
            repmat(reshape(t, 1, 1, n_obs), 2, 2, 1) .* repmat(vars(:, :, x_dim), 1, 1, n_obs));
    end
% %     densityFn = @(x_dim) mvnpdf(y_hmm(:, :, x_dim),...
% %         repmat(t, 1, 2) .* repmat(means(x_dim, :), n_obs, 1),...
% %         repmat(reshape(t, 1, 1, n_obs), 2, 2, 1) .* repmat(vars(:, :, x_dim), 1, 1, n_obs));
% %     emission_densities = cell2mat(arrayfun(densityFn, 1:dim_x, 'UniformOutput', false));
    emission_densities = emission_densities .* ~p0;

end