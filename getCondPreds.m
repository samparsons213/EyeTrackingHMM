function [cond_preds] = getCondPreds(l_dirs, l_dirs_orth, means, vars)
% Transforms the parameters of the latent state conditional Gaussians into
% conditional predictions

% Inputs:

% l_dirs:       2 by n_dirs array of unit vectors in each of the modelled
%               positive latent directions

% l_dirs_orth:  2 by n_dirs array of unit vectors orthogonal to each of the 
%               modelled positive latent directions

% means:        (n_dirs+1) by 2 array of mean vectors, one for each
%               latent state. No movement state correpsonds to first row,
%               following rows correpsond to each direction in l_dirs in
%               turn

% vars:         2 by 2 by (n_dirs+1) array of covariance matrices,
%               one for each latent state

% Outputs:

% cond_preds:   (n_dirs+1) by 2 array of conditional predictions

% Author:       Sam Parsons
% Date created: 14/07/16
% Last amended: 14/07/16

    n_dirs = size(l_dirs, 2);
    cond_preds = [means(1, :); zeros(n_dirs, 2)];
    El_dir = exp(means(2:end, 1) + reshape(vars(1, 1, 2:end), n_dirs, 1)./2);
    cond_preds(2:end, :) = l_dirs' .* repmat(El_dir, 1, 2) +...
        l_dirs_orth' .* repmat(means(2:end, 2), 1, 2);

end