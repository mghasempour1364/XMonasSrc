function [S_star, runtime] = MOGAHHO_main(X_train, y_train, X_val, y_val, N_pop, T_max)
% MOGAHHO_MAIN: Hybrid MOGA + HHO (baseline)
    
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
        
        [~, best_idx] = max(fitness(:, 1));
        v_best = P(best_idx, :);
        
        for i = 1:N_pop
            if rand() < 0.5
                % HHO update
                E = -1 + 2 * rand();
                J = rand();
                P(i, :) = v_best - E * abs(J * v_best - P(i, :));
            else
                % GA crossover
                idx1 = randi(N_pop);
                idx2 = randi(N_pop);
                if fitness(idx1, 1) > fitness(idx2, 1)
                    parent1 = P(idx1, :);
                else
                    parent1 = P(idx2, :);
                end
                crossover_mask = rand(1, d) < 0.5;
                P(i, :) = parent1;
                P(i, crossover_mask) = ~P(i, crossover_mask);
            end
            P(i, :) = P(i, :) > 0.5;
        end
    end
    
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