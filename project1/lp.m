function x = lp(A, b, tol)
% LP Solve Basis Pursuit via linear programing
%
% Solves the following problem:
%   min_x || x ||_1 s.t. b = Ax
%
% The solution is returned in the vector x.


% Converted the problem to LP by using a vector t=[y1 y2 ... ym x1 x2 ... xm]'
% where y1=|x1|, y2=|x2|, ... ym=|xm| and minimizing f't with the new constraints


[n,m] = size(A); %just to use the variables
f = [ones(m,1); zeros(m,1)]; %minimize f't which is equal to ||x||_1
lb = [zeros(m,1); -inf*ones(m,1)]; %the y values must be positive
Aeq = [zeros(n,m) A]; %to formulate A*x = b with the use of t vector
beq = b;
A = [-eye(m) eye(m); -eye(m) -eye(m)]; %since yi>=xi and yi>=-xi
b = zeros(2*m,1);

% for Matlab 2017 use the following 2 lines
options = optimoptions('linprog','Algorithm','dual-simplex','Display','none','OptimalityTolerance',tol);
t = linprog(f, A, b, Aeq, beq, lb, [], options);

% for Matlab 2015 use the following 2 lines
% options = optimoptions('linprog','Algorithm','dual-simplex','Display','none','Tolfun',tol);
% t = linprog(f, A, b, Aeq, beq, lb, [], [], options);

x  = t(m+1:end,1);
x = sparse(x);


end