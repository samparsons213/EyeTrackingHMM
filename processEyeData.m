clear all
close all

% We load the plain data for the eye tracking experiment
load clean_data2

% We process the data to have a sequence of proyected values into the 
% different directions
eye_diff = clean_data2{2}(2:end, :) - clean_data2{2}(1:(end-1), :);

% Load eye movement directions and its orthogonals
load l_dirs

% Process
[y_hmm, p0] = yTransformHMM(eye_diff(:, 1:2), l_dirs);