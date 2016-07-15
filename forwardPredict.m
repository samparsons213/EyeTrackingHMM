function [px_t] = forwardPredict(px_tm1, P)
% Performs the forward predict step of EM, taking the filtered posterior at
% time t-1 and returning the one step ahead latent predictive distribution
% at time t

% Inputs:

% px_tm1:       1 by dim_x filtered posterior P(X_t-1 | Y_{1:t-1})

% P:            dim_x by dim_x transition matrix, with each row containing
%               non-negative numbers summing to 1

% Outputs:

% px_t:         1 by dim_x one step ahead latent predictive distribution
%               P(X_t | Y_{1:t-1})

% Author:       Sam Parsons
% Date created: 03/07/16
% Last amended: 11/07/16

    px_t = px_tm1 * P;

end