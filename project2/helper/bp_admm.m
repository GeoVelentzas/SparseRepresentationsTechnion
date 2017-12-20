function v = bp_admm(CA, b, lambda)
% BP_ADMM Solve Basis Pursuit problem via ADMM
%
% Solves the following problem:
%   min_x 1/2*||b - CAx||_2^2 + lambda*|| x ||_1
%
% The solution is returned in the vector v.
 
% Set the accuracy tolerance of ADMM, run for at most max_admm_iters
tol_admm = 1e-4;
max_admm_iters = 100;
 
% TODO: Compute the vector of inner products between the atoms and the signal
% Write your code here... CAtb = ????;
CAtb = CA'*b;
CAtb = sparse(CAtb);

% In the x-update step of the ADMM we use the Cholesky factorization for
% solving efficiently a given linear system Ax=b. The idea of this
% factorization is to decompose a symmetric positive-definite matrix A
% by A = L*L^T = L*U, where L is a lower triangular matrix and U is
% its transpose. Given L and U, we can solve Ax = b by first solving
% Ly = b for y by forward substitution, and then solving Ux = y
% for x by back substitution.
% To conclude, given A and b, where A is symmetric and positive-definite, 
% we first compute L using Matlab's command L = chol( A, 'lower' );
% and get U by setting U = L'; Then, we obtain x via x = U \ (L \ b);
% Note that the matrix A is fixed along the iterations of the ADMM
% (and so as L and U). Therefore, in order to reduce computations,
% we compute its decomposition once.

% TODO: Compute the Cholesky factorization of M = CA'*CA + I for fast computation 
% of the x-update. Use Matlab's chol function and produce a lower triangular
% matrix L, satisfying the equation M = L*L'
% Write your code here... L = chol( ????, ???? );
L = chol(CA'*CA + eye(size(CA'*CA,1)), 'lower');
 
% Force Matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');
 
% TODO: Initialize v
% Write your code here... v = ????;
v = zeros(size(CA,2),1);
v = sparse(v);

% TODO: Initialize u, the dual variable of ADMM
% Write your code here... u = ????;
u = zeros(size(CA,2),1);
u = sparse(u); 

% TODO: Initialize the previous estimate of v, used for convergence test
% Write your code here... v_prev = ????;
v_prev = v;
 
% main loop
for i = 1:max_admm_iters
 
    % TODO: x-update via Cholesky factorization. Solve the linear system
    % (CA'*CA + I)x = (CAtb + v - u)
    % Write your code here... x = ????    
    x = U\(L\(CAtb + v - u));

    % TODO: v-update via soft thresholding
    % Write your code here... v = ????;
    v = x + u;
    v(v<lambda&v>-lambda) = 0;
    v(v>=lambda) = v(v>=lambda)-lambda;
    v(v<=-lambda) = v(v<=-lambda)+lambda;

    % TODO: u-update according to the ADMM formula
    % Write your code here... u = ????;
    u = u + x - v;
    
    % Check if converged   
    if norm(v) && (norm((v - v_prev)) / norm(v)) < tol_admm
         break;
    end
    
    % TODO: Save the previous estimate in v_prev
    % Write your code here... v_prev = ????;
    v_prev = v;
 
end
 
end

