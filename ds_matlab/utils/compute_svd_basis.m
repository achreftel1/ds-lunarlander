function [mymodes, mymodes2] = compute_svd_basis(X_c, X_w, n_lift_modes, n_obs_modes)
    combined_buffer = [X_c, X_w];
    [Phi, ~]        = svd(combined_buffer, 'econ');
    
    mymodes  = Phi(:, 1:n_lift_modes);  % Lifting nonlinear basis space
    mymodes2 = Phi(:, 1:n_obs_modes);   % Reference projection system surface
end