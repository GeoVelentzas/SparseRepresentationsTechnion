% In this project we will solve a variant of the P0^\epsilon for filling-in 
% missing pixels (also known as "inpainting") in a synthetic image.
 
close all;
clear; clc;

addpath(genpath('./'));

%% Parameters
 
% Set the size of the desired image is (n x n)
n = 40;


% Set the number of atoms
m = 2*40*40;

% Set the percentage of known data
p = 0.4;

% Set the noise std
sigma = 0.05;

% Set the cardinality of the representation
true_k = 10;

% Base seed - A non-negative integer used to reproduce the results
% Set an arbitrary value for base_seed
base_seed = 10;

% Run the different algorithms for num_experiments and average the results
num_experiments = 10;
 
 
%% Create a dictionary A of size (n^2 x m) for Mondrian-like images
 
% initialize A with zeros
A = sparse(n^2,m);
 
% In this part we construct A by creating its atoms one by one, where
% each atom is a rectangle of random size (in the range 5-20 pixels),
% position (uniformly spread in the area of the image), and sign. 
% Lastly we will normalize each atom to a unit norm.
for i=1:size(A,2)

    % Choose a specific random seed to reproduce the results
    rng(i + base_seed);
    
    empty_atom_flag = 1;    
    while empty_atom_flag
        
        % Create a rectangle of random size and position
        init_row = randi(n-5+1);
        init_col = randi(n-5+1);
        final_row = min(init_row + 4 + ceil(20*rand), n);
        final_col = min(init_col + 4 + ceil(20*rand), n);
        random_matrix = zeros(n,n);
        random_matrix(init_row:final_row,init_col:final_col) = sign(randn);
        atom = sparse(random_matrix(:));
                
        % Verify that the atom is not empty or nearly so
        if norm(atom(:)) > 1e-5
            empty_atom_flag = 0;
            
            % Normalize the atom
            atom = atom/norm(atom);
            
            % Assign the generated atom to the matrix A
            A(:,i) = atom(:);
        end
        
    end
    
end
 
%% Oracle Inpainting
 
% Allocate a vector to store the PSNR results
PSNR_oracle = zeros(num_experiments,1);

disp('running oracle...');
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Choose a specific random seed to reproduce the results
    rng(experiment + base_seed);
    
    % Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k);
    
    % Compute the subsampled dictionary
    A_eff = C*A;
    
    % Compute the oracle estimation
    s = find(x0~=0);
    x_oracle = oracle(A_eff, b, s);    
    
    % Compute the estimated image    
    b_oracle = A*x_oracle;

    % Compute the PSNR
    PSNR_oracle(experiment) = compute_psnr(b0, b_oracle);
    disp(['experiment: ', num2str(experiment)]);
end
 
% Display the average PSNR of the oracle
fprintf('Oracle: Average PSNR = %.3f\n', mean(PSNR_oracle));
 
%% Greedy: OMP Inpainting
 
% We will sweep over k = 1 up-to k = max_k and pick the best result
max_k = min(2*true_k, m);
 
% Allocate a vector to store the PSNR estimations per each k
PSNR_omp = zeros(num_experiments,max_k);
 
disp('running omp...');
% Loop over the different realizations
for experiment = 1:num_experiments
    
    % Choose a specific random seed to reproduce the results
    rng(experiment + base_seed);
    
    % Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k);
    
    % Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
   
    % Run the OMP for various values of k and pick the best results
    for k_ind = 1:max_k
        
        % Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, k_ind);
        
        % Un-normalize the coefficients
        x_omp = x_omp./atoms_norm';
        
        % Compute the estimated image        
        b_omp = A*x_omp;
        
        % Compute the current PSNR
        PSNR_omp(experiment, k_ind) = compute_psnr(b0, b_omp);
        
        % Save the best result of this realization, we will present it later
        if PSNR_omp(experiment, k_ind) == max(PSNR_omp(experiment, :))
            best_b_omp = b_omp;
        end
        
    end
    disp(['experiment: ', num2str(experiment)]);
end
 
% Compute the best PSNR, computed for different values of k
PSNR_omp_best_k = max(PSNR_omp,[],2);
 
% Display the average PSNR of the OMP (obtained by the best k per image)
fprintf('OMP: Average PSNR = %.3f\n', mean(PSNR_omp_best_k));
 
% Plot the average PSNR vs. k
psnr_omp_k = mean(PSNR_omp,1);
figure(1); plot(1:max_k, psnr_omp_k, '-*r', 'LineWidth', 2);
ylabel('PSNR [dB]'); xlabel('k'); grid on;
title(['OMP: PSNR vs. k, True Cardinality = ' num2str(true_k)]);
 
 
%% Convex relaxation: Basis Pursuit Inpainting via ADMM
 
% We will sweep over various values of lambda
num_lambda_values = 10;
 
% Allocate a vector to store the PSNR results obtained for the best lambda
PSNR_admm_best_lambda = zeros(num_experiments,1);

disp('running admm...');
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Choose a specific random seed to reproduce the results
    rng(experiment + base_seed);
    
    % Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k);
    
    % Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
    
    % Run the BP for various values of lambda and pick the best result
    lambda_max = norm( A_eff_normalized'*b, 'inf' );
    lambda_vec = logspace(-5,0,num_lambda_values)*lambda_max;    
    psnr_admm_lambda = zeros(1,num_lambda_values);
    
    % Loop over various values of lambda
    for lambda_ind = 1:num_lambda_values
        
        % Compute the BP estimation
        x_admm = bp_admm(A_eff_normalized, b, lambda_vec(lambda_ind));
        
        % Un-normalize the coefficients
        x_admm = x_admm./atoms_norm';
        
        % Compute the estimated image        
        b_admm = A*x_admm;
        
        % Compute the current PSNR
        psnr_admm_lambda(lambda_ind) = compute_psnr(b0, b_admm);
        
        % Save the best result of this realization, we will present it later
        if psnr_admm_lambda(lambda_ind) == max(psnr_admm_lambda)
            best_b_admm = b_admm;
        end
        
    end
    
    % Save the best PSNR
    PSNR_admm_best_lambda(experiment) = max(psnr_admm_lambda);
    disp(['experiment: ', num2str(experiment)]);
    
end
 
% Display the average PSNR of the BP
fprintf('BP via ADMM: Average PSNR = %.3f\n', mean(PSNR_admm_best_lambda));
 
% Plot the PSNR vs. lambda of the last realization
figure(2); semilogx(lambda_vec, psnr_admm_lambda, '-*r', 'LineWidth', 2);
ylabel('PSNR [dB]'); xlabel('\lambda'); grid on;
title('BP via ADMM: PSNR vs. \lambda');
 
%% show the results
 
% Show the images obtained in the last realization, along with their PSNR
figure(3); 
subplot(2,3,1); imagesc(reshape(full(b0),n,n)); 
colormap(gray); axis equal;
title(['Original Image, k = ' num2str(true_k)]);
 
subplot(2,3,2); imagesc(reshape(full(b0_noisy),n,n)); 
colormap(gray); axis equal;
title(['Noisy Image, PSNR = ' num2str(compute_psnr(b0, b0_noisy))]);
 
subplot(2,3,3); imagesc(reshape(full(C'*b),n,n)); 
colormap(gray); axis equal;
title(['Corrupted Image, PSNR = ' num2str(compute_psnr(b0, C'*b))]);
 
subplot(2,3,4); imagesc(reshape(full(b_oracle),n,n)); 
colormap(gray); axis equal;
title(['Oracle, PSNR = ' num2str(compute_psnr(b0, b_oracle))]);
 
subplot(2,3,5); imagesc(reshape(full(best_b_omp),n,n)); 
colormap(gray); axis equal;
title(['OMP, PSNR = ' num2str(compute_psnr(b0, best_b_omp))]);
 
subplot(2,3,6); imagesc(reshape(full(best_b_admm),n,n));
colormap(gray); axis equal;
title(['BP-ADMM, PSNR = ' num2str(compute_psnr(b0, best_b_admm))]);
 
%% Compare the results

% Show a bar plot of the average PSNR value obtained per each algorithm
figure(4);
mean_psnr = [mean(PSNR_oracle) mean(PSNR_omp_best_k) mean(PSNR_admm_best_lambda)];
bar(mean_psnr);
set(gca,'XTickLabel',{'Oracle','OMP','BP-ADMM'});
ylabel('PSNR [dB]'); xlabel('Algorithm');
 
%% Run OMP with fixed cardinality and increased percentage of known data
 
% Set the noise std
sigma = 0.05;

% Set the cardinality of the representation
true_k = 5;

% Create a vector of increasing values of p in the range [0.4 1]. The
% length of this vector equal to num_values_of_p = 7.
num_values_of_p = 7;
p_vec = linspace(0.4,1,7);

% We will repeat the experiment for num_experiments realizations
num_experiments = 100;
 
% Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_p = zeros(num_values_of_p,1);
 
disp('running omp with fixed cardinality and persentage of known data...');
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Loop over various values of p
    for p_ind = 1:num_values_of_p
        
        % Choose a specific random seed to reproduce the results
        rng(experiment + base_seed);
        
        % Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p_vec(p_ind), sigma, true_k);
                
        % Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
        
        % Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k);
        
        % Un-normalize the coefficients
        x_omp = x_omp./atoms_norm';
        
        % Compute the estimated image        
        b_omp = A*x_omp;
                
        % Compute the MSE of the estimate
        cur_mse = 1/40^2*norm(b0-b_omp)^2;
                
        % Compute the current normalized MSE and aggregate
        mse_omp_p(p_ind) = mse_omp_p(p_ind) + cur_mse / noise_std^2;
    end
    
    disp(['experiment: ', num2str(experiment), ' from 100']);
    
end
 
% Compute the average PSNR over the different realizations
mse_omp_p = mse_omp_p / num_experiments;
 
% Plot the average normalized MSE vs. p
figure(5); plot(p_vec, mse_omp_p, '-*r', 'LineWidth', 2);
ylabel('Normalized-MSE'); xlabel('p'); grid on;
title(['OMP with k = ' num2str(true_k) ', Normalized-MSE vs. p'])
 
 
%% Run OMP with fixed cardinality and increased noise level
 
% Set the cardinality of the representation
true_k = 5;

% Set the percentage of known data
p = 0.5;

% Create a vector of increasing values of sigma in the range [0.15 0.5].
% The length of this vector equal to num_values_of_sigma = 10.
num_values_of_sigma = 10;
sigma_vec = linspace(0.15, 0.5, num_values_of_sigma);


% Repeat the experiment for num_experiments realizations
num_experiments = 100;
 
% Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_sigma = zeros(num_values_of_sigma,1);

disp('running omp with fixed cardinality and increasing noise...');
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Loop over increasing noise level
    for sigma_ind = 1:num_values_of_sigma
        
        % Choose a specific random seed to reproduce the results
        rng(experiment + base_seed);
        
        % Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma_vec(sigma_ind), true_k);
        
        % Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
        
        % Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k);
        
        % Un-normalize the coefficients
        x_omp = x_omp./atoms_norm';
        
        % Compute the estimated image        
        b_omp = A*x_omp;
                
        % Compute the MSE of the estimate
        % Write your code here... cur_mse = ????;
        cur_mse = 1/40^2*norm(b0-b_omp)^2;
        
        % Compute the current normalized MSE and aggregate
        mse_omp_sigma(sigma_ind) = mse_omp_sigma(sigma_ind) + cur_mse / noise_std^2;
 
    end
    disp(['experiment: ', num2str(experiment), ' from 100']);
end
 
% Compute the average PSNR over the different realizations
mse_omp_sigma = mse_omp_sigma / num_experiments;
    
% Plot the average normalized MSE vs. sigma
figure(6); plot(sigma_vec, mse_omp_sigma, '-*r', 'LineWidth', 2);
ylim([0.5*min(mse_omp_sigma) 5*max(mse_omp_sigma)]);
ylabel('Normalized-MSE'); xlabel('sigma'); grid on;
title(['OMP with k = ' num2str(true_k) ', Normalized-MSE vs. sigma']);

