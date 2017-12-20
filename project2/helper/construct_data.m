function [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, k)
% CONSTRUCT_DATA Generate Mondrian-like synthetic image
%
% Input:
%  A     - Dictionary of size (n^2 x m)
%  p     - Percentage of known data in the range (0 1]
%  sigma - Noise std
%  k     - Cardinality of the representation of the synthetic image in the
%          range [1 max(m,n)]
%
% Output:
%  x0 - A sparse vector creating the Mondrian-like image b0
%  b0 - The original image of size n^2
%  noise_std  - The standard deviation of the noise added to b0
%  b0_noisy   - A noisy version of b0 of size n^2
%  C  - Sampling matrix of size (p*n^2 x n^2), 0 < p <= 1
%  b  - The corrupted image (noisy and subsampled version of b0) of size p*n^2
 
 
% Get the size of the image and number of atoms
[n_squared, m] = size(A);
n = sqrt(n_squared);
 
%% generate a Mondrian image
%  by drawing at random a sparse vector x0 of length m with cardinality k
 
% Draw at random the locations of the non-zeros
nnz_locs = randperm(m);
nnz_locs = nnz_locs(1:k);
 
% Draw at random the values of the coefficients
nnz_vals = randn(1,k);
 
% Create a k-sparse vector x0 of length m given the nnz_locs and nnz_vals
x0=sparse(m,1);
x0(nnz_locs,1) = nnz_vals;
 
% Given A and x0, compute the signal b0
b0 = A*x0;

 
%% Create the measured data vector b of size n^2
 
% Compute the dynamic range
dynamic_range = max(b0)-min(b0);
 
% Create a noise vector
noise_std = sigma*dynamic_range;
noise = noise_std*randn(n^2,1);
 
% Add noise to the original image
b0_noisy = b0 + noise;
 
 
%% Create the sampling matrix C of size (p*n^2 x n^2), 0 < p <= 1
 
% Create an identity matrix of size (n^2 x n^2)
I = eye(n^2);
 
% Draw at random the indices of rows to be kept
inds = randperm(n^2);
keep_inds = inds(1:round(p*n^2));
 
% Create the sampling matrix C of size (p*n^2 x n^2) by keeping rows
% from I that correspond to keep_inds
C = sparse(I(keep_inds,:));

 
% Create a subsampled version of the noisy image
b = C*b0_noisy;
 
end
 
