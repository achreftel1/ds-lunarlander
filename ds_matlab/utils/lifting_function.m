function g = lifting_function(n_lift_modes)
    sym_vars = sym('a', [n_lift_modes, 1]);
    vec_idx = 1;
    lift_poly_vec = zeros((n_lift_modes * (n_lift_modes + 1))/2, 1);
    lift_poly_vec = sym(lift_poly_vec);

    for row = 1:n_lift_modes
        for col = 1:row
            lift_poly_vec(vec_idx, 1) = sym_vars(row) * sym_vars(col);
            vec_idx = vec_idx + 1;
        end
    end
    g = matlabFunction(lift_poly_vec);
end