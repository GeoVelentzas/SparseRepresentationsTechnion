% In this project we demonstrate the OMP and BP algorithms, by running them 
% on a set of signals and checking whether they provide the desired outcome
 
%% Parameters

addpath(genpath('./'));

% Set the length of the signal
n = 50;

% Set the number of atoms in the dictionary
m = 100;

% Set the maximum number of non-zeros in the generated vector
s_max = 15;

% Set the minimal entry value
min_coeff_val = 1;

% Set the maximal entry value
max_coeff_val = 3;

% Number of realizations
num_realizations = 200;

% Base seed: A non-negative integer used to reproduce the results
% Set an arbitrary value for base seed
base_seed = 13;


%% Create the dictionary
 
% Create a random matrix A of size (n x m)
rng(10); %to have the same A matrix... remove as you may wish
A = randn(n,m);

% Normalize the columns of the matrix to have a unit norm
A_normalized = A./(ones(n,1)*sqrt(sum(A.^2)));


%% Create data and run OMP and BP
 
% Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4;
% Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4;
 
% Allocate a matrix to save the L2 error of the obtained solutions
L2_error = zeros(s_max,num_realizations,2); 
% Allocate a matrix to save the support recovery score
support_error = zeros(s_max,num_realizations,2);
           
% Loop over the sparsity level
disp('Please Wait...')
for s = 1:s_max
    
    % Use the same random seed in order to reproduce the results if needed
    rng(s+base_seed)
    
    % Loop over the number of realizations
    for experiment = 1:num_realizations
   
        % In this part we will generate a test signal b = A_normalized*x by 
        % drawing at random a sparse vector x with s non-zeros entries in 
        % true_supp locations with values in the range of [min_coeff_val, max_coeff_val]
        x = zeros(m,1);
        
        % Draw at random a true_supp vector
        true_supp = randperm(m, s)';
        
        % Draw at random the coefficients of x in true_supp locations
        x(true_supp) = ((max_coeff_val-min_coeff_val)*rand(s,1)+min_coeff_val).*sign(randn(s,1));        
        
        % Create the signal b
        b = A_normalized*x;
        
        % Run OMP
        x_omp = omp(A_normalized, b, s);
                
        % Compute the relative L2 error
        L2_error(s, experiment, 1) = norm(x_omp-x)^2/norm(x)^2;
        
        % Get the indices of the estimated support
        estimated_supp = find(x_omp~=0);
        
        % Compute the support recovery score
        support_error(s, experiment, 1) = 1 - length(intersect(estimated_supp, true_supp))/max(length(estimated_supp), length(true_supp));
        
        % Run BP
        x_lp = lp(A_normalized, b, tol_lp);
        x_lp(abs(x_lp)<=eps_coeff)= 0;
        
        % Compute the relative L2 error
        L2_error(s,experiment,2) = norm(x_lp-x)^2/norm(x)^2;
        
        % Get the indices of the estimated support, where the
        % coeffecients are larger (in absolute value) than eps_coeff
        estimated_supp = find(x_lp~=0); 
        
        % Compute the support recovery error
        support_error(s, experiment, 2) = 1 - length(intersect(estimated_supp, true_supp))/max(length(estimated_supp), length(true_supp));
        
        fprintf('Running... %0.1f %% \n', ((s-1)*(num_realizations) + experiment)/(s_max*num_realizations)*100) %remove line if error occurs
        
    end
    
end
 
%% Display the results 
 
% Plot the average relative L2 error, obtained by the OMP and BP versus the cardinality
figure(1); clf; 
plot(1:s_max,mean(L2_error(1:s_max,:,1),2),'r','LineWidth',2); hold on;
plot(1:s_max,mean(L2_error(1:s_max,:,2),2),'g','LineWidth',2); 
xlabel('Cardinality of the true solution');
ylabel('Average and Relative L_2-Error');
set(gca,'FontSize',14);
legend({'OMP','LP'});
axis([0 s_max 0 1]);
 
% Plot the average support recovery score, obtained by the OMP and BP versus the cardinality
figure(2); clf; 
plot(1:s_max,mean(support_error(1:s_max,:,1),2),'r','LineWidth',2); hold on;
plot(1:s_max,mean(support_error(1:s_max,:,2),2),'g','LineWidth',2); 
xlabel('Cardinality of the true solution');
ylabel('Probability of Error in Support');
set(gca,'FontSize',14);
legend({'OMP','LP'});
axis([0 s_max 0 1]);
