function [uni, pwise] = getLatentPosteriorMarginals(y_hmm, t, pi1, P,...
    means, vars, p0)
% Returns the univariate and pairwise marginals of the latent posterior 
% given the data and current parameter values

% Inputs:

% y_hmm:        n_obs by 2 (n_dirs+1) array of transformed difference
%               vectors for each of the latent directions. First element in
%               3rd dimension is the untransformed data y corresponding to
%               the no movement latent state. The following elements of the
%               3rd dimension correspond to each row of l_dirs, in order.

% t:            n_obs by 1 integer array giving the time passed 
%               corresponding to each difference vector

% pi1:          1 by dim_x probability vector for latent prior at t=1

% P:            dim_x by dim_x transition matrix

% means:        (n_dirs+1) by 2 array of mean vectors, one for each
%               latent state

% vars:         2 by 2 by (n_dirs+1) array of covariance matrices,
%               one for each latent state

% p0:           n_obs by (n_dirs+1) logical array giving true when the 
%               probability of a latent direction given an observed 
%               difference vector is zero (i.e. the dot product between the
%               vector and the latent direction is non-positive for 
%               movement states). First col is always false as all 
%               observations are logically consistent with target staying 
%               still

% Outputs:

% uni:          T by dim_x array of univariate latent marginals 
%               P(X_t | Y_{1:T})

% pwise:        dim_x by dim_x by (T-1) array of pairwise latent marginals
%               P(X_t, X_t+1 | Y_{1:T})

% Author:       Sam Parsons
% Date created: 12/07/16
% Last amended: 12/07/16

    T = size(y_hmm, 1);
    dim_x = length(pi1);
    emission_densities = getEmissionDensities(y_hmm, t, means, vars, p0);
    uni = [forwardUpdate(pi1, emission_densities(1, :)); zeros(T-1, dim_x)];
    pwise = zeros(dim_x, dim_x, T-1);
    for t = 2:T
%         fprintf('Forward pass: time = %d of %d\n', t, T)
% %         if t >= 140
% %             disp('hello')
% %         end
        uni(t, :) = forwardPredict(uni(t-1, :), P);
        uni(t, :) = forwardUpdate(uni(t, :), emission_densities(t, :));
    end
    for t = (T-1):-1:1
%         fprintf('Backward pass: time = %d\n', t)
        pwise(:, :, t) = backwardPredict(uni(t, :), P);
        [pwise(:, :, t), uni(t, :)] =...
            backwardUpdate(pwise(:, :, t), uni(t+1, :));
    end

end