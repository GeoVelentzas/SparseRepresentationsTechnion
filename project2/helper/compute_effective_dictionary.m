function [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)
%% COMPUTE_EFFECTIVE_DICTIONARY Computes the subsampled and normalized
%   dictionary
%
% Input:
%  C     - Sampling matrix of size (p*n^2 x n^2)
%  A     - Dictionary of size (n^2 x m)
%
% Output:
%  A_eff_normalized - The subsampled and normalized dictionary of size (p*n^2 x m)
%  atoms_norm - A vector of length m, containing the norm of each sampled atom
 
 
% Compute the subsampled dictionary
A_eff = C*A;
 
% Compute the norm of each atom
atoms_norm = sqrt(sum(A_eff.^2,1));
 
% Normalize the columns of A_eff, avoid division by zero
atoms_norm(atoms_norm == 0) = 1e-5;
A_eff_normalized = A_eff./(ones(size(A_eff,1),1)*atoms_norm);


 
end

