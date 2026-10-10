% ==================== FILE 2:  utils/evaluate_lander.m ===================
function [FPR, TPR, threshold_steps, dE_c, dE_w, totaltokens] = evaluate_lander( ...
              data_dir, test_start, test_end, mymodes, mymodes2, g, A_c, A_w, ...
              n_obs_modes, tokens_length_threshold, threshold_steps)
    % Same maths as the original evaluate.m (Eq. 5-8 of the paper), with:
    %   - a configurable eta sweep range (lander residuals are small)
    %   - dE_c / dE_w returned so we can plot histograms and compute F1
    %   - totaltokens taken as the length of EACH trajectory separately

    total_test = test_end - test_start + 1;
    HH_c = zeros(1, total_test); NN_c = zeros(1, total_test);
    HH_w = zeros(1, total_test); NN_w = zeros(1, total_test);
    totaltokens = zeros(1, total_test);
    eval_idx = 1;

    zero_cell  = num2cell(zeros(size(mymodes, 2), 1));
    dummy_lift = g(zero_cell{:});
    lifted_total_dim = n_obs_modes + length(dummy_lift);

    for k = test_start:test_end
        filename = fullfile(data_dir, sprintf('sample_%d.mat', k));
        if ~exist(filename, 'file'), continue; end

        dataObj     = matfile(filename);
        correct_emb = dataObj.correct_embeddings;
        wrong_emb   = dataObj.wrong_embeddings;

        buf_c    = mymodes2.' * correct_emb;   buf_w    = mymodes2.' * wrong_emb;
        buf_calt = mymodes.'  * correct_emb;   buf_walt = mymodes.'  * wrong_emb;

        correct_emb_lift = zeros(lifted_total_dim, size(correct_emb, 2));
        wrong_emb_lift   = zeros(lifted_total_dim, size(wrong_emb, 2));

        for j = 1:size(correct_emb, 2)
            cell_c = num2cell(buf_calt(:, j));
            lift_c = g(cell_c{:});
            correct_emb_lift(:, j) = [buf_c(:, j); lift_c];
        end
        for j = 1:size(wrong_emb, 2)
            cell_w = num2cell(buf_walt(:, j));
            lift_w = g(cell_w{:});
            wrong_emb_lift(:, j) = [buf_w(:, j); lift_w];
        end

        % Frobenius norm of the residual matrix == sqrt( sum_k eps_k^2 ),
        % i.e. exactly the two terms of Eq. (7).
        HH_c(eval_idx) = norm(A_w * correct_emb_lift(:, 1:end-1) - mymodes2.' * correct_emb(:, 2:end), 'fro');
        NN_c(eval_idx) = norm(A_c * correct_emb_lift(:, 1:end-1) - mymodes2.' * correct_emb(:, 2:end), 'fro');
        HH_w(eval_idx) = norm(A_w * wrong_emb_lift(:,   1:end-1) - mymodes2.' * wrong_emb(:,   2:end), 'fro');
        NN_w(eval_idx) = norm(A_c * wrong_emb_lift(:,   1:end-1) - mymodes2.' * wrong_emb(:,   2:end), 'fro');

        % NOTE: the original code stored only size(wrong_emb,2) here and then
        % applied that single length filter to BOTH classes. We keep the
        % shorter of the two, which is the conservative choice.
        totaltokens(eval_idx) = min(size(correct_emb, 2), size(wrong_emb, 2));
        eval_idx = eval_idx + 1;
    end

    n = eval_idx - 1;
    HH_c = HH_c(1:n); NN_c = NN_c(1:n);
    HH_w = HH_w(1:n); NN_w = NN_w(1:n);
    totaltokens = totaltokens(1:n);

    % Eq. (7):  dE = ||eps_h|| - ||eps_c||   ( > 0 expected for correct )
    dE_c = HH_c - NN_c;      % correct trajectories
    dE_w = HH_w - NN_w;      % wrong trajectories

    num_steps = length(threshold_steps);
    FPR = cell(1, length(tokens_length_threshold));
    TPR = cell(1, length(tokens_length_threshold));

    for qq = 1:length(tokens_length_threshold)
        idx  = find(totaltokens >= tokens_length_threshold(qq));
        dc   = dE_c(idx);  dw = dE_w(idx);
        FPR{qq} = zeros(1, num_steps);
        TPR{qq} = zeros(1, num_steps);
        s = 1;
        for eta = threshold_steps
            fp = sum(dc <  eta);   tn = sum(dc >= eta);
            tp = sum(dw <  eta);   fn = sum(dw >= eta);
            FPR{qq}(s) = fp / max(fp + tn, 1);
            TPR{qq}(s) = tp / max(tp + fn, 1);
            s = s + 1;
        end
        fprintf('     K >= %3d : %d test pairs\n', tokens_length_threshold(qq), length(idx));
    end
end