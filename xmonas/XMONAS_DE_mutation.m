function new_pos = XMONAS_DE_mutation(P, i, F, CR)
% XMONAS_DE_MUTATION: Differential Evolution mutation (Eq. 19)
%
% Inputs:
%   P   : Current population (N_pop x d)
%   i   : Index of the particle to mutate
%   F   : Mutation scaling factor
%   CR  : Crossover rate
%
% Output:
%   new_pos : Mutated particle position

    N_pop = size(P, 1);
    d = size(P, 2);
    
    % Select three random distinct particles
    idx = randperm(N_pop, 3);
    while any(idx == i)
        idx = randperm(N_pop, 3);
    end
    
    v_r = P(idx(1), :);
    v_s = P(idx(2), :);
    v_t = P(idx(3), :);
    
    % Mutation
    mutant = v_r + F * (v_s - v_t);
    
    % Crossover
    crossover_mask = rand(1, d) < CR;
    new_pos = P(i, :);
    new_pos(crossover_mask) = mutant(crossover_mask);
end