function [px_t] = forwardUpdate(px_tm1, emission_densities)
% Performs the forward update step of EM, taking the one step ahead latent
% predictive posterior at time t-1 and returning the latent posterior
% at time t

% Inputs:

% px_tm1:               1 by dim_x one step ahead latent predictive
%                       distribution P(X_t | Y_{1:t-1})

% emission_densities:   1 by dim_x array if conditional emission densities
%                       given each latent state

% Outputs:

% px_t:                 1 by dim_x latent posterior P(X_t | Y_{1:t})

% Author:               Sam Parsons
% Date created:         03/07/16
% Last amended:         11/07/16

    px_t =...
        (px_tm1 .* emission_densities) ./ (px_tm1 * emission_densities');

end