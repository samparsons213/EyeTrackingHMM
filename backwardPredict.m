function [px_tgtp1] = backwardPredict(px_t, P)
% Performs the backward predict step of EM, taking the filtered posterior
% at time t and the transition matrix and returning the 
% filtered posterior only further conditioned on x at time t+1

% Inputs:

% px_t:         1 by dim_x filtered posterior P(X_t | Y_{1:t})

% px_tp1:       1 by dim_x smoothed posterior P(X_t+1 | Y_{1:T})

% P:            dim_x by dim_x transition matrix, with each row containing
%               non-negative numbers summing to 1

% Outputs:

% px_tgtp1:     dim_x by dim_x one step ahead latent predictive 
%               distribution P(X_t | Y_{1:t-1}, X_t+1). Each row
%               corresponds to a distribution conditioned on a specific
%               value of X_t+1, and therefore sums to 1

% Author:       Sam Parsons
% Date created: 12/07/16
% Last amended: 12/07/16

    dim_x = length(px_t);
    px_tgtp1 = repmat(px_t, dim_x, 1) .* P';
    px_tgtp1 = px_tgtp1 ./ repmat(sum(px_tgtp1, 2), 1, dim_x);

end