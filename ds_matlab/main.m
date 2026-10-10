% ======================================
%  Koopman-based hallucination detection
%  ======================================
clear; close all; clc;

% import functions
addpath('utils');

% 1. Initialization
DATA_DIR     = 'data_path_to_embeddings_LLM_responses';
EMB_DIM      = 1024;  
TRAIN_LIMIT  = 2000;  
N_LIFT_MODES = 40;   
N_OBS_MODES  = 500;  
TEST_START   = 2000;
TEST_END     = 9999;
tokens_length_threshold = [1 50 100 150];

% 2. Call functions for hallucination detection pipeline
fprintf('===> Step 1: Load training data\n');
[X_c, X_c_plus, X_w, X_w_plus] = load_training_data(DATA_DIR, EMB_DIM, TRAIN_LIMIT);

fprintf('===> Step 2: Compute data basis\n');
[mymodes, mymodes2] = compute_svd_basis(X_c, X_w, N_LIFT_MODES, N_OBS_MODES);

fprintf('===> Step 3: Construct lifted state\n');
g = lifting_function(N_LIFT_MODES);

% Lifting initialization
zero_cell = num2cell(zeros(N_LIFT_MODES, 1));
dummy_eval = g(zero_cell{:});
lifted_dim = N_OBS_MODES + length(dummy_eval);

% Allocate full state space size
X_c_full      = zeros(lifted_dim, size(X_c, 2));
X_c_plus_full = zeros(lifted_dim, size(X_c, 2));
X_w_full      = zeros(lifted_dim, size(X_w, 2));
X_w_plus_full = zeros(lifted_dim, size(X_w, 2));

fprintf('===> Step 4: Construct data matrices for dynamical system fitting\n');
for j = 1:size(X_c, 2)
    cell_c = num2cell(mymodes.' * X_c(:, j));
    cell_c_plus = num2cell(mymodes.' * X_c_plus(:, j));
    
    lift_c = g(cell_c{:});
    lift_c_plus = g(cell_c_plus{:});
    
    X_c_full(:, j)      = [mymodes2.' * X_c(:, j); lift_c];
    X_c_plus_full(:, j) = [mymodes2.' * X_c_plus(:, j); lift_c_plus];
end

for j = 1:size(X_w, 2)
    cell_w = num2cell(mymodes.' * X_w(:, j));
    cell_w_plus = num2cell(mymodes.' * X_w_plus(:, j));
    
    lift_w = g(cell_w{:});
    lift_w_plus = g(cell_w_plus{:});
    
    X_w_full(:, j)      = [mymodes2.' * X_w(:, j); lift_w];
    X_w_plus_full(:, j) = [mymodes2.' * X_w_plus(:, j); lift_w_plus];
end

fprintf('===> Step 5: Run DMD\n');
% Fit the dynamical system models
[A_c, A_w] = fit_dynamical_systems(X_c_full, X_c_plus_full, X_w_full, X_w_plus_full, N_OBS_MODES);

fprintf('===> Step 6: Evaluate on test set\n');
[FPR, TPR, threshold_steps] = evaluate(...
    DATA_DIR, TEST_START, TEST_END, mymodes, mymodes2, g, A_c, A_w, N_OBS_MODES, tokens_length_threshold);

% 3. Visualization
fprintf('===> Step 7: Plot ROC curves\n');
figure;
legend_labels = cell(1, length(tokens_length_threshold));
line_styles   = {'-', '--', ':', '-.'};

for k = 1:length(tokens_length_threshold)
    semilogx(FPR{k}, TPR{k}, line_styles{k}, 'Color', 'g', 'LineWidth', 1.5);
    hold on;
    legend_labels{k} = sprintf('$L \\ge %d$', tokens_length_threshold(k));
end

grid on; axis square;
xlabel('False Positive Rate (FPR)', 'Interpreter', 'latex'); 
ylabel('True Positive Rate (TPR)', 'Interpreter', 'latex');
title('ROC Curves for different sequence lengths $L$', 'Interpreter', 'latex');
legend(legend_labels, 'Location', 'southeast', 'Interpreter', 'latex');
