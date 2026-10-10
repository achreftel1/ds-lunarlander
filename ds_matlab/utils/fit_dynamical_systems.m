function [A_c, A_w] = fit_dynamical_systems(X_c_full, X_c_plus_full, X_w_full, X_w_plus_full, n_obs_modes)
    % Koopman Operator linear regression (DMD).
    % Computes global transition dynamics matrices from lifted state representations
    % and isolates target observable subspace operators.

    % Perform Dynamic Mode Decomposition (DMD) regression via pseudo-inverse
    A_w_full = X_w_plus_full * pinv(X_w_full);
    A_c_full = X_c_plus_full * pinv(X_c_full);

    % Truncate and isolate transition mappings targeting only observable baseline states
    A_w   = A_w_full(1:n_obs_modes, :);
    A_c   = A_c_full(1:n_obs_modes, :);
end