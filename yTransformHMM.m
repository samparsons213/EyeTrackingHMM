function [y_hmm, p0] = yTransformHMM(y, l_dirs)
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

%% Initialization
% Create space for the final output
n_dirs = size(l_dirs, 2);
y_hmm = cat(3, y, zeros([size(y), n_dirs]));

%% Calculate distance travelled in each latent direction at each 
%  movement as dot product of each difference vector with each (unit)
%  latent direction
n_obs = size(y, 1);
% Inner product and transform to tensor
d1 = reshape(y * l_dirs, [n_obs, 1, n_dirs]);
% Determine which latent directions have a valid positive projection
% (i.e. dot product is strictly positive)
p0 = d1 <= 0;
y_hmm(:, 1, 2:end) = log(d1) .* ~p0;
y_hmm(isnan(y_hmm)) = 0;

%% Calculate distance travelled in the orthogonal direction to each latent direction

% Substract the projection and calculate the norm to know the distance
% in the orthogonal direction from the base direction
proj_diff =  bsxfun(@minus, y, bsxfun(@times, d1, reshape(l_dirs, [1, 2, n_dirs]))); 
proj_diff_norm = sqrt(sum(proj_diff.^2, 2));
y_hmm(:, 2, 2:end) = proj_diff_norm .* ~p0;

p0 = [false(n_obs, 1), squeeze(p0)];

end