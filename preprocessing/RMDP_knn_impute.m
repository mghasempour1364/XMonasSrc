function [X_train_imp, X_test_imp, params] = RMDP_knn_impute(X_train, X_test, K)
% RMDP_KNN_IMPUTE: KNN imputation for missing values (Eq. 2)
%
% Inputs:
%   X_train : Training data (N_train x d) with possible NaN values
%   X_test  : Test data (N_test x d) with possible NaN values
%   K       : Number of nearest neighbors
%
% Outputs:
%   X_train_imp : Imputed training data
%   X_test_imp  : Imputed test data
%   params      : Struct with imputation parameters

    [N_train, d] = size(X_train);
    [N_test, ~] = size(X_test);
    
    % Copy data
    X_train_imp = X_train;
    X_test_imp = X_test;
    
    % Identify missing values in training data
    missing_train = isnan(X_train);
    
    % If no missing values, return directly
    if ~any(missing_train(:))
        params.K = K;
        params.train_mean = mean(X_train, 1, 'omitnan');
        return;
    end
    
    % --- Impute training data using KNN ---
    for i = 1:N_train
        missing_idx = find(missing_train(i, :));
        if isempty(missing_idx)
            continue;
        end
        
        % Find K nearest neighbors (using non-missing features)
        available_idx = setdiff(1:d, missing_idx);
        if isempty(available_idx)
            % If all features are missing, use column mean
            X_train_imp(i, missing_idx) = mean(X_train(:, missing_idx), 1, 'omitnan');
            continue;
        end
        
        % Compute distances to all other training samples
        dists = zeros(N_train, 1);
        for j = 1:N_train
            if j == i
                dists(j) = inf;
                continue;
            end
            diff_vec = X_train(i, available_idx) - X_train(j, available_idx);
            dists(j) = norm(diff_vec(~isnan(diff_vec)));
        end
        
        % Find K nearest neighbors
        [~, sorted_idx] = sort(dists, 'ascend');
        neighbor_idx = sorted_idx(1:min(K, N_train-1));
        
        % Impute missing features using weighted average of neighbors
        for f = missing_idx
            neighbor_vals = X_train(neighbor_idx, f);
            neighbor_vals = neighbor_vals(~isnan(neighbor_vals));
            if isempty(neighbor_vals)
                X_train_imp(i, f) = mean(X_train(:, f), 'omitnan');
            else
                X_train_imp(i, f) = mean(neighbor_vals);
            end
        end
    end
    
    % --- Impute test data using training data as reference ---
    missing_test = isnan(X_test);
    for i = 1:N_test
        missing_idx = find(missing_test(i, :));
        if isempty(missing_idx)
            continue;
        end
        
        available_idx = setdiff(1:d, missing_idx);
        if isempty(available_idx)
            X_test_imp(i, missing_idx) = mean(X_train(:, missing_idx), 1, 'omitnan');
            continue;
        end
        
        % Compute distances to all training samples
        dists = zeros(N_train, 1);
        for j = 1:N_train
            diff_vec = X_test(i, available_idx) - X_train(j, available_idx);
            dists(j) = norm(diff_vec(~isnan(diff_vec)));
        end
        
        [~, sorted_idx] = sort(dists, 'ascend');
        neighbor_idx = sorted_idx(1:min(K, N_train));
        
        for f = missing_idx
            neighbor_vals = X_train(neighbor_idx, f);
            neighbor_vals = neighbor_vals(~isnan(neighbor_vals));
            if isempty(neighbor_vals)
                X_test_imp(i, f) = mean(X_train(:, f), 'omitnan');
            else
                X_test_imp(i, f) = mean(neighbor_vals);
            end
        end
    end
    
    % Store parameters
    params.K = K;
    params.train_mean = mean(X_train_imp, 1, 'omitnan');
end