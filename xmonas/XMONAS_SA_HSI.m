function [S_star, pareto_archive] = XMONAS_SA_HSI(S_NS, X_train, y_train, X_val, y_val, ...
    N_pop, T_max, lambda, delta, CR, F_range)
% XMONAS_SA_HSI: Self-Adaptive Hybrid Swarm Intelligence (Section 3.3)

    [~, d] = size(X_train);
    
    % Default parameters
    if nargin < 6, N_pop = 30; end
    if nargin < 7, T_max = 100; end
    if nargin < 8, lambda = [0.4, 0.3, 0.3]; end
    if nargin < 9, delta = 0.5; end
    if nargin < 10, CR = 0.9; end
    if nargin < 11, F_range = [0.5, 1.0]; end
    
    % --- Initialize population (binary vectors) ---
    P = rand(N_pop, d) > 0.5;
    
    % --- Initialize Pareto archive (empty struct array with defined fields) ---
    pareto_archive = struct('subset', {}, 'fitness', {});
    
    % --- Main loop ---
    for t = 1:T_max
        % --- Compute fitness for each particle ---
        fitness = zeros(N_pop, 3);
        for i = 1:N_pop
            F_i = XMONAS_fitness(P(i, :), X_train, y_train, X_val, y_val, S_NS);
            fitness(i, :) = F_i;
        end
        
        % --- Update Pareto archive ---
        pareto_archive = update_pareto_archive(pareto_archive, P, fitness);
        
        % --- Compute population diversity D(t) ---
        if N_pop > 1
            D_t = mean(pdist(P, 'hamming'));
        else
            D_t = 0;
        end
        
        % --- Compute convergence G_best(t) ---
        if t == 1
            G_best_prev = min(fitness(:, 1));
        end
        G_best = min(fitness(:, 1));
        
        % --- Compute Shannon entropy H(t) ---
        p_j = mean(P, 1);
        H_t = -sum(p_j .* log(p_j + 1e-10));
        
        % --- Compute switching probabilities (Eq. 15-16) ---
        Psi_t = lambda(1) * D_t + lambda(2) * (1 - abs(G_best - G_best_prev) / max(G_best, 1e-10)) + lambda(3) * H_t;
        Prob_HHO = 1 / (1 + exp(-Psi_t));
        Prob_CPO = 1 - Prob_HHO;
        Prob_DE = delta * H_t;
        
        % --- Normalize probabilities ---
        total_prob = Prob_HHO + Prob_CPO + Prob_DE;
        if total_prob == 0
            total_prob = 1;
        end
        Prob_HHO = Prob_HHO / total_prob;
        Prob_CPO = Prob_CPO / total_prob;
        Prob_DE = Prob_DE / total_prob;
        
        % --- Update each particle ---
        for i = 1:N_pop
            r = rand();
            if r < Prob_HHO
                P(i, :) = XMONAS_HHO_update(P, i, fitness);
            elseif r < Prob_HHO + Prob_CPO
                P(i, :) = XMONAS_CPO_update(P, i, fitness);
            else
                F = F_range(1) + (F_range(2) - F_range(1)) * rand();
                P(i, :) = XMONAS_DE_mutation(P, i, F, CR);
            end
        end
        
        % --- Apply binary transfer function (thresholding) ---
        P = P > 0.5;
        
        % --- Check stagnation ---
        if mod(t, 10) == 0 && abs(G_best - G_best_prev) < 1e-6
            Prob_HHO = min(Prob_HHO + 0.1, 1.0);
        end
        
        G_best_prev = G_best;
    end
    
    % --- Select final subset from Pareto front ---
    S_star = select_from_pareto(pareto_archive, d);
end

% --- Helper function: Update Pareto archive ---
function archive = update_pareto_archive(archive, P, fitness)
    for i = 1:size(P, 1)
        is_dominated = false;
        for j = 1:length(archive)
            if all(archive(j).fitness <= fitness(i, :)) && any(archive(j).fitness < fitness(i, :))
                is_dominated = true;
                break;
            end
        end
        if ~is_dominated
            new_entry = struct('subset', P(i, :), 'fitness', fitness(i, :));
            if isempty(archive)
                archive = new_entry;
            else
                archive(end+1) = new_entry;
            end
        end
    end
    archive = remove_dominated(archive);
end

% --- Helper function: Remove dominated solutions ---
function archive = remove_dominated(archive)
    if isempty(archive)
        return;
    end
    keep = true(length(archive), 1);
    for i = 1:length(archive)
        for j = 1:length(archive)
            if i ~= j && all(archive(j).fitness <= archive(i).fitness) && any(archive(j).fitness < archive(i).fitness)
                keep(i) = false;
                break;
            end
        end
    end
    archive = archive(keep);
end

% --- Helper function: Select from Pareto archive (FINAL, ERROR-FREE) ---
function S_star = select_from_pareto(archive, d)
    % If archive is empty, return all features
    if isempty(archive) || length(archive) == 0
        S_star = true(1, d);
        return;
    end
    
    % Find solution with highest accuracy (F1) using a simple loop
    best_idx = 1;
    best_acc = archive(1).fitness(1);
    
    for i = 2:length(archive)
        if archive(i).fitness(1) > best_acc
            best_acc = archive(i).fitness(1);
            best_idx = i;
        end
    end
    
    % Return the corresponding subset
    S_star = archive(best_idx).subset;
end