## EyeTrackingHMM

Modelling eye tracking data using an HMM

Before files are described a quick mention of missing data problem is needed. 
As we discussed in the last meeting, sometimes a large number of rows are missing between data points. 
As a quick hack to overcome this, I modified our original model so the conditional Gaussians for each latent state are actually 
a sum of i.i.d. Gaussians, one for each time step in the difference vector. 
For example, if the difference in time stamps indicates 2 missing rows, 
the conditional emission density for each latent state would be N(3*mu, 3*Sigma) for some mean vector mu and covariance matrix Sigma

# Contents

Quick description of all files:


clean_data2.dat:	The original eye-tracking data. It's a MATLAB data file, load into MATLAB using load('clean_data2.dat', '-MAT'). 
                    Only object is a cell with the time-stamps in the first element and the data (T by 3 array of reals). 
                    You probably won't need this as we're using the difference vectors of the data


y_hmm.mat:	The difference vectors of the original data. This is also a MATLAB data file, load using load('y_hmm.mat'). 
            Contains a few variables: 
                y_hmm:  n_obs by 3 by 9 array of difference vectors, transformed to be of the form 
                        for each latent state (Third dimension is latent state index, state 1 is no movement, 
                        states 2-9 go clockwise from north through to northwest). Data for latent state 1 is untransformed, 
                        for the other states the 2d vector at each timepoint has as first element the log-distance 
                        in the latent direction (if this distance is non-positive then there will be a 0 in both elements) 
                        and in the second element is the (signed) distance travelled in orthogonal direction latent direction
	
                t:  n_obs by 1 integer array of differences in timestamps. All entries are divisble by 4, 
                    a value of 8 indicates 1 missing row of original data, a value of 16 indicates 3 missing rows etc
	
                p0:	n_obs by 9 logical array where true at (x,y) indicates that at time x there is 0 probability of latent state 
                    y being active
	
                n_obs: integer giving number of rows in y_hmm (58761)
	
                dim_x:	integer giving number of latent states (9)
	
                dim_y:	integer giving size of each observation (2)

	
l_dirs.mat:	This is a MATLAB data file with 2 variables:
	l_dirs: 2 by 9 array of unit vectors in each latent direction
	l_dirs_orth: 2 by 9 array of unit vectors orthogonal to each latent direction


theta0.mat:	This is a MATLAB data file containing some randomly intialised model parameters:
	means:	9 by 2 array of mean vectors for each latent state
	vars:	2 by 2 by 9 array of covariance matrices for each latent state
	pi1:	1 by 9 probability vector for latent state at time t=1
	P:		9 by 9 matrix of conditional transition probabilities, P(i, j) = prob(transition from state i to state j)

theta_new.mat:	This is a MATLAB data file containing some fitted parameters, similar to theta0.mat

yTransformHMM.m:    MATLAB function for transforming original difference vectors into the form needed for each latent state. 
                    Takes the original difference vectors, the latent direction unit vectors, and the orthogonal 
                    unit vectors as arguments. 
                    
	
secondEM.m:	MATLAB function that performs EM for a specified number of iterations or until convergence. 
            Each iteration consists of finding latent posteriors and then maximising the likelihood


getLatentPosteriorMarginals.m:	MATLAB function that uses forward-backward algorithm to obtain latent posterior distributions. 
                                Forward pass is 2 stage pass from time t=1 to t=n_obs, backward pass is 2 stage pass from time t=n_obs to t=1.


forwardPredict.m	MATLAB function that completes first stage of forward pass


forwardUpdate.m:	MATLAB function that completes second stage of forward pass


backwardPredict.m:	MATLAB function that completes first stage of backward pass


backwardUpdate.m:	MATLAB function that completes second stage of backward pass


getEmissionDensities:	MATLAB function that returns the conditional emission densities for the transformed observations for each latent state


maximiseParameters:	MATLAB function that returns the likelihood maximising parameters for a given latent posterior


thresholdCovMatrix.m	MATLAB function that thresholds the eigenvalues of an estimated covariance matrix to ensure it is postive definite.


getCondPreds.m	MATLAB function that returns the predicted difference vectors given each latent state for a given set of parameters