function [pi1, P, means, vars] = secondEM(y_hmm, t, pi1, P, means, vars,...
    p0, epsilon, em_iters, min_eig)
% Performs EM updates on initial HMM parameters, stopping after either
% converge (norm(theta - theta_old) < epsilon) or completion of em_iters
% iterations

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

% epsilon:      small positive real number, change threshold for
%               convergence

% em_iters:     large positive integer, max number of em iterations to
%               perform

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

    conv = false;
    for em_iter = 1:em_iters
        fprintf('Iteration: %d,\t', em_iter)
        theta_old = [pi1(:); P(:); means(:); vars(:)];
        [uni, pwise] = getLatentPosteriorMarginals(y_hmm, t, pi1, P, means,...
            vars, p0);
        [pi1, P, means, vars] = maximiseParameters(uni, pwise, y_hmm, t,...
            min_eig);
        theta = [pi1(:); P(:); means(:); vars(:)];
        delta = norm(theta - theta_old);
        fprintf('delta = %f\n', delta)
        if delta < epsilon
            conv = true;
            fprintf('Algorithm terminated due to parameter convergence\n')
            break
        end
    end
    if ~conv
        fprintf('Algorithm terminated due to reaching maximum number of iterations\n')
    end

end