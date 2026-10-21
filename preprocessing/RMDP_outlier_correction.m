function [X_train_out, X_test_out, params] = RMDP_outlier_correction(X_train, X_test, tau)
% RMDP_OUTLIER_CORRECTION: Z-score based outlier detection and correction (Eq. 3)
%
% Inputs:
%   X_train : Training data (N_train x d)
%   X_test  : Test data (N_test x d)
%   tau     : Z-score threshold (typically 3)
%
% Outputs:
%   X_train_out : Outlier-corrected training data
%   X_test_out  : Outlier-corrected test data
%   params      : Struct with mean and std computed on training data

    % Compute mean and std on training data only
    mu = mean(X_train, 1, 'omitnan');
    sigma = std(X_train, 0, 1, 'omitnan');
    
    % Prevent division by zero
    sigma(sigma == 0) = 1;
    
    % Copy data
    X_train_out = X_train;
    X_test_out = X_test;
    
    % --- Correct outliers in training data ---
    for j = 1:size(X_train, 2)
        z = (X_train(:, j) - mu(j)) / sigma(j);
        outlier_idx = find(abs(z) > tau);
        if ~isempty(outlier_idx)
            X_train_out(outlier_idx, j) = mu(j) + tau * sigma(j) * sign(z(outlier_idx));
        end
    end
    
    % --- Correct outliers in test data using training statistics ---
    for j = 1:size(X_test, 2)
        z = (X_test(:, j) - mu(j)) / sigma(j);
        outlier_idx = find(abs(z) > tau);
        if ~isempty(outlier_idx)
            X_test_out(outlier_idx, j) = mu(j) + tau * sigma(j) * sign(z(outlier_idx));
        end
    end
    
    % Store parameters
    params.mu = mu;
    params.sigma = sigma;
    params.tau = tau;
end