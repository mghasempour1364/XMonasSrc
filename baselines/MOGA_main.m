function [S_star, runtime] = MOGA_main(X_train, y_train, X_val, y_val, N_pop, T_max)
% MOGA_MAIN: Multi-Objective Genetic Algorithm (baseline)
    
    tic;
    [~, d] = size(X_train);
    P = rand(N_pop, d) > 0.5;
    
    for t = 1:T_max
        fitness = zeros(N_pop, 2);
        for i = 1:N_pop
            selected_idx = find(P(i, :));
            if isempty(selected_idx)
                fitness(i, :) = [0, d];
            else
                try
                    mdl = fitcknn(X_train(:, selected_idx), y_train, 'NumNeighbors', 5);
                    y_pred = predict(mdl, X_val(:, selected_idx));
                    fitness(i, 1) = mean(y_pred == y_val);
                catch
                    fitness(i, 1) = 0;
                end
                fitness(i, 2) = sum(P(i, :));
            end
        end
        
        % Selection, crossover, mutation
        new_P = P;
        for i = 1:N_pop
            idx1 = randi(N_pop);
            idx2 = randi(N_pop);
            if fitness(idx1, 1) > fitness(idx2, 1)
                parent1 = P(idx1, :);
            else
                parent1 = P(idx2, :);
            end
            idx3 = randi(N_pop);
            idx4 = randi(N_pop);
            if fitness(idx3, 1) > fitness(idx4, 1)
                parent2 = P(idx3, :);
            else
                parent2 = P(idx4, :);
            end
            
            crossover_mask = rand(1, d) < 0.5;
            child = parent1;
            child(crossover_mask) = parent2(crossover_mask);
            
            mutation_mask = rand(1, d) < 0.01;
            child(mutation_mask) = ~child(mutation_mask);
            
            new_P(i, :) = child;
        end
        P = new_P;
    end
    
    % Return best subset
    final_fitness = zeros(N_pop, 1);
    for i = 1:N_pop
        selected_idx = find(P(i, :));
        if ~isempty(selected_idx)
            try
                mdl = fitcknn(X_train(:, selected_idx), y_train, 'NumNeighbors', 5);
                y_pred = predict(mdl, X_val(:, selected_idx));
                final_fitness(i) = mean(y_pred == y_val);
            catch
                final_fitness(i) = 0;
            end
        end
    end
    [~, best_idx] = max(final_fitness);
    S_star = P(best_idx, :);
    runtime = toc;
end