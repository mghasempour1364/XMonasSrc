function new_pos = XMONAS_CPO_update(P, i, fitness)
% XMONAS_CPO_UPDATE: Competitive Population Optimization update rule (Eq. 18)
%
% Inputs:
%   P       : Current population (N_pop x d)
%   i       : Index of the particle to update
%   fitness : Fitness matrix (N_pop x 3)
%
% Output:
%   new_pos : Updated particle position

    N_pop = size(P, 1);
    r = rand();
    
    % Select superior and inferior particles
    [~, sorted_idx] = sort(fitness(:, 1), 'descend');
    superior_idx = sorted_idx(1:ceil(N_pop/4));
    inferior_idx = sorted_idx(end-ceil(N_pop/4)+1:end);
    
    v_j = P(superior_idx(randi(length(superior_idx))), :);
    v_k = P(inferior_idx(randi(length(inferior_idx))), :);
    v_i = P(i, :);
    
    new_pos = v_i + r * (v_j - v_k);
end