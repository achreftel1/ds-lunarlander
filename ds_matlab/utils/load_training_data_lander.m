% =========================================================================
%  Save the TWO functions below as TWO separate files inside utils/ :
%     utils/load_training_data_lander.m
%     utils/evaluate_lander.m
%  (MATLAB requires one function per file, named after the file.)
% =========================================================================


% ============== FILE 1:  utils/load_training_data_lander.m ===============
function [X_c, X_c_plus, X_w, X_w_plus] = load_training_data_lander(data_dir, emb_dim, train_limit)
    % Same logic as the original load_training_data.m, with two fixes for
    % the lander:
    %   (1) MAX_BUFFER_SIZE raised: lander episodes are ~100-900 steps,
    %       so 500 episodes is ~250k transitions, not ~100k.
    %   (2) a progress print, because the loop is longer.

    MAX_BUFFER_SIZE = 2000000;          % was 100000

    alldata_c     = zeros(emb_dim, MAX_BUFFER_SIZE);
    alldataplus_c = zeros(emb_dim, MAX_BUFFER_SIZE);
    alldata_w     = zeros(emb_dim, MAX_BUFFER_SIZE);
    alldataplus_w = zeros(emb_dim, MAX_BUFFER_SIZE);

    sample_idx  = 1;
    d_c = 0; d_w = 0;
    train_count = 0;

    while train_count < train_limit
        filename = fullfile(data_dir, sprintf('sample_%d.mat', sample_idx));
        if sample_idx > train_limit * 10
            error('ran out of sample files at index %d', sample_idx);
        end
        if exist(filename, 'file')
            matObj      = matfile(filename);
            correct_emb = matObj.correct_embeddings;   % 8 x K_c
            wrong_emb   = matObj.wrong_embeddings;     % 8 x K_w

            % snapshot pairs are built PER TRAJECTORY, so no pair ever
            % crosses an episode boundary
            n_c = size(correct_emb, 2) - 1;
            alldata_c(:,     d_c+1 : d_c+n_c) = correct_emb(:, 1:end-1);
            alldataplus_c(:, d_c+1 : d_c+n_c) = correct_emb(:, 2:end);
            d_c = d_c + n_c;

            n_w = size(wrong_emb, 2) - 1;
            alldata_w(:,     d_w+1 : d_w+n_w) = wrong_emb(:, 1:end-1);
            alldataplus_w(:, d_w+1 : d_w+n_w) = wrong_emb(:, 2:end);
            d_w = d_w + n_w;

            train_count = train_count + 1;
            if mod(train_count, 100) == 0
                fprintf('     loaded %d/%d episodes  (%d c-trans, %d h-trans)\n', ...
                        train_count, train_limit, d_c, d_w);
            end
        end
        sample_idx = sample_idx + 1;
    end

    X_c      = alldata_c(:,     1:d_c);
    X_c_plus = alldataplus_c(:, 1:d_c);
    X_w      = alldata_w(:,     1:d_w);
    X_w_plus = alldataplus_w(:, 1:d_w);
end