function new_pos = XMONAS_HHO_update(P, i, fitness)
% XMONAS_HHO_UPDATE: Harris Hawks Optimization update rule (Eq. 17)
%
% Inputs:
%   P       : Current population (N_pop x d)
%   i       : Index of the particle to update
%   fitness : Fitness matrix (N_pop x 3)
%
% Output:
%   new_pos : Updated particle position

    [~, best_idx] = max(fitness(:, 1));
    v_best = P(best_idx, :);
    v_i = P(i, :);
    
    E = -1 + 2 * rand(); % Escape energy factor (Eq. 17)
    J = rand();         % Jump strength
    
    new_pos = v_best - E * abs(J * v_best - v_i);
end