function [FPR, TPR, threshold_steps] = evaluate(...
              data_dir, test_start, test_end, mymodes, mymodes2, g, A_c, A_w, n_obs_modes, tokens_length_threshold)

    total_test_samples = test_end - test_start + 1;
    HH_c = zeros(1, total_test_samples); NN_c = zeros(1, total_test_samples);
    HH_w = zeros(1, total_test_samples); NN_w = zeros(1, total_test_samples);
    totaltokens = zeros(1, total_test_samples);
    eval_idx = 1;
    
    % Safely unpack dummy cell array on an isolated line
    zero_cell = num2cell(zeros(size(mymodes, 2), 1));
    dummy_lift = g(zero_cell{:});
    lifted_total_dim = n_obs_modes + length(dummy_lift);

    for k = test_start:test_end
        filename = fullfile(data_dir, sprintf('sample_%d.mat', k));
        if ~exist(filename, 'file'), continue; end
        
        dataObj = matfile(filename);
        correct_emb = dataObj.correct_embeddings;
        wrong_emb   = dataObj.wrong_embeddings;
        
        buf_c = mymodes2.' * correct_emb;   buf_w = mymodes2.' * wrong_emb;
        buf_calt = mymodes.' * correct_emb; buf_walt = mymodes.' * wrong_emb;
        
        correct_emb_lift = zeros(lifted_total_dim, size(correct_emb, 2));
        wrong_emb_lift   = zeros(lifted_total_dim, size(wrong_emb, 2));
        
        % Process Factual Matrix Trajectories
        for j = 1:size(correct_emb, 2)
            cell_c = num2cell(buf_calt(:, j));
            % FIX: Isolate function evaluation from matrix concatenation
            lift_val_c = g(cell_c{:}); 
            correct_emb_lift(:, j) = [buf_c(:, j); lift_val_c];
        end
        
        % Process Hallucinated Matrix Trajectories
        for j = 1:size(wrong_emb, 2)
            cell_w = num2cell(buf_walt(:, j));
            % FIX: Isolate function evaluation from matrix concatenation
            lift_val_w = g(cell_w{:}); 
            wrong_emb_lift(:, j)   = [buf_w(:, j); lift_val_w];
        end
        
        HH_c(eval_idx) = norm(A_w * correct_emb_lift(:, 1:end-1) - mymodes2.' * correct_emb(:, 2:end), 'fro');
        NN_c(eval_idx) = norm(A_c * correct_emb_lift(:, 1:end-1) - mymodes2.' * correct_emb(:, 2:end), 'fro');
        HH_w(eval_idx) = norm(A_w * wrong_emb_lift(:, 1:end-1) - mymodes2.' * wrong_emb(:, 2:end), 'fro');
        NN_w(eval_idx) = norm(A_c * wrong_emb_lift(:, 1:end-1) - mymodes2.' * wrong_emb(:, 2:end), 'fro');
        
        totaltokens(eval_idx) = size(wrong_emb, 2);
        eval_idx = eval_idx + 1;
    end
    
    % Boundary Decision Metrics Sweep Configuration
    threshold_steps  = -250 : 0.04 : 250;
    num_steps        = length(threshold_steps);
    
    FPR = cell(1, length(tokens_length_threshold));
    TPR = cell(1, length(tokens_length_threshold));
    
    for qq = 1:length(tokens_length_threshold)
        valid_indices = find(totaltokens >= tokens_length_threshold(qq));
        HH_wbuf = HH_w(valid_indices); NN_cbuf = NN_c(valid_indices);
        HH_cbuf = HH_c(valid_indices); NN_wbuf = NN_w(valid_indices);
        
        FPR{qq} = zeros(1, num_steps); 
        TPR{qq} = zeros(1, num_steps);
        
        step_idx = 1;
        for jjj = threshold_steps
            current_fp = sum((NN_cbuf - HH_cbuf + jjj) > 0);
            current_tn = sum((NN_cbuf - HH_cbuf + jjj) < 0);
            current_tp = sum((HH_wbuf - NN_wbuf - jjj) < 0);
            current_fn = sum((HH_wbuf - NN_wbuf - jjj) > 0);
            
            FPR{qq}(step_idx) = current_fp / (current_fp + current_tn);
            TPR{qq}(step_idx) = current_tp / (current_tp + current_fn);
            step_idx = step_idx + 1;
        end
    end
end
