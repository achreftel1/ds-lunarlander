% =========================================================================
%  main_lander.m
%  Koopman-based DS detection  --  LunarLander version
%
%  Same pipeline as main.m from the hallucination-detection-DS repo.
%  Only the dimensions change, because the "embedding" here is the 8-dim
%  physical state of the lander instead of a token embedding:
%
%      x_k = (x, y, vx, vy, angle, ang_vel, leg1, leg2)  in R^8
%
%  correct_embeddings  <-> safe-landing trajectory   (regime c)
%  wrong_embeddings    <-> crash trajectory          (regime h / w)
%
%  Place this file next to main.m (so that addpath('utils') resolves).
%  Generate the data first with:  python prepare_lander_for_ds.py
% =========================================================================
clear; close all; clc;

addpath('utils');

% ---------------------------------------------------------------- 1. config
DATA_DIR     = 'results/ds_data';   % folder of sample_k.mat written by the python script
EMB_DIM      = 8;                   % <-- lander state dimension (was 1024)
TRAIN_LIMIT  = 500;                 % episodes used to FIT  (was 2000 LLM responses)
N_LIFT_MODES = 8;                   % cannot exceed EMB_DIM  (was 40)
N_OBS_MODES  = 8;                   % cannot exceed EMB_DIM  (was 500)
TEST_START   = 501;                 % must be > TRAIN_LIMIT
TEST_END     = 5000;
tokens_length_threshold = [1 100 200 300];   % episode-length filters (was [1 50 100 150])

% lander residuals are MUCH smaller than LLM-embedding residuals,
% so the eta sweep range from evaluate.m (-250:0.04:250) is far too wide.
THRESHOLD_STEPS = -20 : 0.002 : 20;

fprintf('EMB_DIM=%d  N_OBS=%d  N_LIFT=%d  train=%d  test=%d..%d\n', ...
        EMB_DIM, N_OBS_MODES, N_LIFT_MODES, TRAIN_LIMIT, TEST_START, TEST_END);

% ---------------------------------------------------- 2. Phase 1: load data
fprintf('===> Step 1: Load training data\n');
[X_c, X_c_plus, X_w, X_w_plus] = load_training_data_lander(DATA_DIR, EMB_DIM, TRAIN_LIMIT);
fprintf('     correct transitions: %d\n', size(X_c, 2));
fprintf('     wrong   transitions: %d\n', size(X_w, 2));

fprintf('===> Step 2: Compute data basis\n');
[mymodes, mymodes2] = compute_svd_basis(X_c, X_w, N_LIFT_MODES, N_OBS_MODES);
% With EMB_DIM = 8 and N_OBS_MODES = 8, mymodes2 is 8x8 orthogonal:
% the projection is a pure rotation and discards nothing.

fprintf('===> Step 3: Construct lifted state\n');
g = lifting_function(N_LIFT_MODES);     % all pairwise products a_i*a_j, j<=i
zero_cell  = num2cell(zeros(N_LIFT_MODES, 1));
dummy_eval = g(zero_cell{:});
lifted_dim = N_OBS_MODES + length(dummy_eval);
fprintf('     lifted dimension = %d + %d = %d\n', ...
        N_OBS_MODES, length(dummy_eval), lifted_dim);

fprintf('===> Step 4: Construct data matrices for dynamical system fitting\n');
X_c_full      = zeros(lifted_dim, size(X_c, 2));
X_c_plus_full = zeros(lifted_dim, size(X_c, 2));
X_w_full      = zeros(lifted_dim, size(X_w, 2));
X_w_plus_full = zeros(lifted_dim, size(X_w, 2));

for j = 1:size(X_c, 2)
    cell_c      = num2cell(mymodes.' * X_c(:, j));
    cell_c_plus = num2cell(mymodes.' * X_c_plus(:, j));
    lift_c      = g(cell_c{:});
    lift_c_plus = g(cell_c_plus{:});
    X_c_full(:, j)      = [mymodes2.' * X_c(:, j);      lift_c];
    X_c_plus_full(:, j) = [mymodes2.' * X_c_plus(:, j); lift_c_plus];
end

for j = 1:size(X_w, 2)
    cell_w      = num2cell(mymodes.' * X_w(:, j));
    cell_w_plus = num2cell(mymodes.' * X_w_plus(:, j));
    lift_w      = g(cell_w{:});
    lift_w_plus = g(cell_w_plus{:});
    X_w_full(:, j)      = [mymodes2.' * X_w(:, j);      lift_w];
    X_w_plus_full(:, j) = [mymodes2.' * X_w_plus(:, j); lift_w_plus];
end

fprintf('===> Step 5: Run DMD\n');
[A_c, A_w] = fit_dynamical_systems(X_c_full, X_c_plus_full, X_w_full, X_w_plus_full, N_OBS_MODES);
fprintf('     A_c is %dx%d,  A_w is %dx%d\n', size(A_c, 1), size(A_c, 2), size(A_w, 1), size(A_w, 2));

% --------------------------------------------- 3. Phase 2: evaluate on test
fprintf('===> Step 6: Evaluate on test set\n');
[FPR, TPR, threshold_steps, dE_c, dE_w, lens] = evaluate_lander( ...
    DATA_DIR, TEST_START, TEST_END, mymodes, mymodes2, g, A_c, A_w, ...
    N_OBS_MODES, tokens_length_threshold, THRESHOLD_STEPS);

% ------------------------------------------------------------ 4. ROC curves
fprintf('===> Step 7: Plot ROC curves\n');
figure;
legend_labels = cell(1, length(tokens_length_threshold));
line_styles   = {'-', '--', ':', '-.'};
for k = 1:length(tokens_length_threshold)
    semilogx(FPR{k}, TPR{k}, line_styles{k}, 'Color', 'g', 'LineWidth', 1.5); hold on;
    auc = abs(trapz(FPR{k}, TPR{k}));
    legend_labels{k} = sprintf('$K \\ge %d$ (AUC %.3f)', tokens_length_threshold(k), auc);
    fprintf('     K >= %3d :  AUC = %.4f\n', tokens_length_threshold(k), auc);
end
grid on; axis square;
xlabel('False Positive Rate (FPR)', 'Interpreter', 'latex');
ylabel('True Positive Rate (TPR)', 'Interpreter', 'latex');
title('LunarLander DS detection: ROC vs episode length $K$', 'Interpreter', 'latex');
legend(legend_labels, 'Location', 'southeast', 'Interpreter', 'latex');

% ------------------------------- 5. Delta-E histogram (paper Fig. 6 style)
figure;
edges = linspace(min([dE_c dE_w]), max([dE_c dE_w]), 60);
histogram(dE_c, edges, 'FaceColor', [0.2 0.7 0.3], 'FaceAlpha', 0.6); hold on;
histogram(dE_w, edges, 'FaceColor', [0.85 0.3 0.3], 'FaceAlpha', 0.6);
xline(0, 'k--', 'LineWidth', 1.2);
xlabel('$\Delta\mathcal{E} = \|\epsilon_h\|_F - \|\epsilon_c\|_F$', 'Interpreter', 'latex');
ylabel('count');
legend({'safe landing (c)', 'crash (h)', '\eta = 0'}, 'Location', 'northwest');
title('Response-level differential residual score, Eq. (7)');

% ---------------------------------------- 6. best threshold, F1, bal. acc.
% positive class = hallucination / crash,   predict crash when dE < eta
best_f1 = -1; best_ba = -1; best_eta_f1 = 0; best_eta_ba = 0;
for eta = threshold_steps
    tp = sum(dE_w <  eta);  fn = sum(dE_w >= eta);
    fp = sum(dE_c <  eta);  tn = sum(dE_c >= eta);
    prec = tp / max(tp + fp, 1);
    rec  = tp / max(tp + fn, 1);
    f1   = 2 * prec * rec / max(prec + rec, eps);
    ba   = 0.5 * (rec + tn / max(tn + fp, 1));
    if f1 > best_f1, best_f1 = f1; best_eta_f1 = eta; end
    if ba > best_ba, best_ba = ba; best_eta_ba = eta; end
end
fprintf('\n--------------------------------------------------------\n');
fprintf('  mean dE | safe landing : %+8.4f\n', mean(dE_c));
fprintf('  mean dE | crash        : %+8.4f\n', mean(dE_w));
fprintf('  best F1       = %5.1f %%  at eta = %+.4f\n', 100 * best_f1, best_eta_f1);
fprintf('  best bal.acc. = %5.1f %%  at eta = %+.4f\n', 100 * best_ba, best_eta_ba);
fprintf('  accuracy at eta = 0 : %5.1f %%\n', ...
        100 * (sum(dE_c >= 0) + sum(dE_w < 0)) / (length(dE_c) + length(dE_w)));
fprintf('--------------------------------------------------------\n');

save('results/lander_ds_results.mat', 'A_c', 'A_w', 'mymodes', 'mymodes2', ...
     'dE_c', 'dE_w', 'lens', 'FPR', 'TPR', 'threshold_steps');
