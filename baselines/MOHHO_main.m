function [S_star, runtime] = MOHHO_main(X_train, y_train, X_val, y_val, N_pop, T_max)
% MOHHO_MAIN: Multi-Objective Harris Hawks Optimization (baseline)
    
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
            E = -1 + 2 * rand();
            J = rand();
            P(i, :) = v_best - E * abs(J * v_best - P(i, :));
            P(i, :) = P(i, :) > 0.5;
        end
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