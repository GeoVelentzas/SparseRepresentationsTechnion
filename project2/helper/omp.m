function x = omp(CA, b, k)
% OMP Solve the sparse coding problem via OMP
%
% Solves the following problem:
%   min_x ||b - CAx||_2^2 s.t. ||x||_0 <= k
%
% The solution is returned in the vector x

% Initialize the vector x
x = zeros(size(CA,2),1);

% Implement the OMP algorithm
r = b;
S = [];
m = size(CA,2);
E = zeros(1, m);
for i = 1:k
    for j = 1:m
        z_opt = CA(:,j)'*r;
        E(j) = norm(r)^2 - norm(z_opt)^2;
    end
    [~, i0] = min(E);
    S = union(S,i0); %since no double entries occur.. else use union(S, i0)
    CAs = CA(:,S);
    xk = (CAs'*CAs)\CAs'*b;
    x(S) = xk;
    r = b - CA*x;
x = sparse(x);




end

