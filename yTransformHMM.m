function [y_hmm, p0] = yTransformHMM(y, l_dirs, l_dirs_orth)
% Takes the difference vectors of dims 1 & 2 of eye tracking data, and
% transforms into the vectors that need to be modelled in each of the
% latent states. For example, the latent state corresponding to strictly
% vertical upwards eye movement models the log upward movement and the
% horizontal movement together as a 2 dimensional Gaussian. l_dirs contains
% the unit vectors in all modelled positive latent directions. In addition 
% there is the no movement state, in which the untransformed vector is 
% modelled as a 2 dimensional Gaussian.

% Inputs:

% y:            n_obs by 2 array of difference vectors for dimensions 1 and
%               2 of eye tracking data

% l_dirs:       2 by n_dirs array of unit vectors in each of the modelled
%               positive latent directions

% l_dirs_orth:  2 by n_dirs array of unit vectors orthogonal to each of the 
%               modelled positive latent directions

% Outputs:

% y_hmm:        n_obs by 2 (n_dirs+1) array of transformed difference
%               vectors for each of the latent directions. First element in
%               3rd dimension is the untransformed data y corresponding to
%               the no movement latent state. The following elements of the
%               3rd dimension correspond to each row of l_dirs, in order.

% p0:           n_obs by (n_dirs+1) logical array giving true when the
%               probability of a latent direction given an observed
%               difference vector is zero (i.e. the dot product between the
%               vector and the latent direction is non-positive for
%               movement states). First col is always false as all
%               observations are logically consistent with target staying
%               still

% Author:       Sam Parsons
% Date created: 11/07/16
% Last amended: 11/07/16

    n_dirs = size(l_dirs, 2);
    y_hmm = cat(3, y, zeros([size(y), n_dirs]));
%     Calculate distance travelled in each latent direction at each 
%     movement as dot product of each difference vector with each (unit)
%     latent direction
    n_obs = size(y, 1);
    d1 = reshape(y * l_dirs, n_obs, 1, n_dirs);
%     Determine which latent directions have a valid positive projection
%     (i.e. dot product is strictly positive)
    p0 = d1 <= 0;
    y_hmm(:, 1, 2:end) = log(d1) .* ~p0;
    y_hmm(isnan(y_hmm)) = 0;
%   Calculate distance travelled in orthogonal direction to each latent
%   direction
    if any(sum(l_dirs .* l_dirs_orth, 1) ~= 0)
        error('at least one vector in l_dirs_orth is not orthogonal to its corresponding vector in l_dirs')
    end
    l_dirs_orth = reshape(l_dirs_orth, 1, 2, n_dirs);
    l_dirs_orth_dim = zeros(n_dirs, 1);
    for dir = 1:n_dirs
        if l_dirs_orth(1, 1, dir) == 0
            l_dirs_orth_dim(dir) = 2;
            l_dirs_orth(1, 1, dir) = l_dirs_orth(1, 2, dir);
        else
            l_dirs_orth_dim(dir) = 1;
        end
    end
    l_dirs_orth = l_dirs_orth(1, 1, :);
    temp = repmat(y, 1, 1, n_dirs) -...
        repmat(d1, 1, 2, 1) .* repmat(reshape(l_dirs, 1, 2, n_dirs), n_obs, 1, 1);
    for obs = 1:n_obs
        for dir = 1:n_dirs
            if ~p0(obs, dir)
                y_hmm(obs, 2, dir+1) =...
                    temp(obs, l_dirs_orth_dim(dir), dir) / l_dirs_orth(dir);
            end
        end
    end
    p0 = [false(n_obs, 1), reshape(p0, n_obs, n_dirs)];

end