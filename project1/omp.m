function x = omp(A, b, k)
% OMP Solve the P0 problem via OMP
%
% Solves the following problem:
%   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
%
% The solution is returned in the vector x

% Initialize the vector x
x = zeros(size(A,2),1);
% for larger applications use sparse function from the first place...

% Implement the OMP algorithm
r = b;
S = [];
m = size(A,2);
E = zeros(1, m);
for i = 1:k
    for j = 1:m
        z_opt = A(:,j)'*r;
        E(j) = norm(r)^2 - norm(z_opt)^2;
    end
    [~, i0] = min(E);
    S = union(S,i0);
    As = A(:,S);
    xk = pinv(As)*b;
    x(S) = xk;
    r = b - A*x;
x = sparse(x);
end
