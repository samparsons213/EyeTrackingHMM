function [px_ttp1, px_t] = backwardUpdate(px_tgtp1, px_tp1)
% Performs the backward update step of EM, taking the filtered posterior
% (with further conditioning from backwardPredict.m) at time t and the
% smoothed posterior at time t+1 and returning the amoothed posterior at 
% time t

% Inputs:

% px_tgtp1:     dim_x by dim_x one step ahead latent predictive 
%               distribution P(X_t | Y_{1:t-1}, X_t+1). Each row
%               corresponds to a distribution conditioned on a specific
%               value of X_t+1, and therefore sums to 1

% px_tp1:       1 by dim_x smoothed posterior P(X_t+1 | Y_{1:T})

% Outputs:

% px_ttp1:      dim_x by dim_x probability array giving 
%               P(X_t, X_t+1 | Y_{1:T}), where the row gives the X_t
%               state and the column give the X_t+1 state

% px_t:         1 by dim_x smoothed posterior at time t 
%               P(X_t | Y_{1:T})

% Author:       Sam Parsons
% Date created: 12/07/16
% Last amended: 12/07/16

    dim_x = length(px_tp1);
    px_ttp1 = repmat(px_tp1, dim_x, 1) .* px_tgtp1';
    px_t = sum(px_ttp1, 2)';

end