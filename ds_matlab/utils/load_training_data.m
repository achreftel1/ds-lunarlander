function [X_c, X_c_plus, X_w, X_w_plus] = load_training_data(data_dir, emb_dim, train_limit)
    % Pre-allocation parameters to handle performance safety bounds
    MAX_BUFFER_SIZE = 100000;
    alldata_c      = zeros(emb_dim, MAX_BUFFER_SIZE);
    alldataplus_c  = zeros(emb_dim, MAX_BUFFER_SIZE);
    alldata_w      = zeros(emb_dim, MAX_BUFFER_SIZE);
    alldataplus_w  = zeros(emb_dim, MAX_BUFFER_SIZE);

    sample_idx = 1;
    d_c = 0; d_w = 0;
    train_count = 0;

    while train_count < train_limit
        filename = fullfile(data_dir, sprintf('sample_%d.mat', sample_idx));
        
        if exist(filename, 'file')
            matObj = matfile(filename);
            correct_emb = matObj.correct_embeddings;
            wrong_emb   = matObj.wrong_embeddings;
            
            % Nominal trajectory shift matching pairs
            num_tokens_c = size(correct_emb, 2) - 1;
            alldata_c(:, d_c+1 : d_c+num_tokens_c)     = correct_emb(:, 1:end-1);
            alldataplus_c(:, d_c+1 : d_c+num_tokens_c) = correct_emb(:, 2:end);
            d_c = d_c + num_tokens_c;
            
            % Hallucinated trajectory shift matching pairs
            num_tokens_w = size(wrong_emb, 2) - 1;
            alldata_w(:, d_w+1 : d_w+num_tokens_w)     = wrong_emb(:, 1:end-1);
            alldataplus_w(:, d_w+1 : d_w+num_tokens_w) = wrong_emb(:, 2:end);
            d_w = d_w + num_tokens_w;
            
            train_count = train_count + 1;
        end
        sample_idx = sample_idx + 1;
    end

    % Strictly truncate bounds to active values
    X_c      = alldata_c(:, 1:d_c);
    X_c_plus = alldataplus_c(:, 1:d_c);
    X_w      = alldata_w(:, 1:d_w);
    X_w_plus = alldataplus_w(:, 1:d_w);
end