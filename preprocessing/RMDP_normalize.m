function [X_train_norm, X_test_norm, params] = RMDP_normalize(X_train, X_test)
% RMDP_NORMALIZE: Min-Max normalization (Eq. 4)
%
% Inputs:
%   X_train : Training data (N_train x d)
%   X_test  : Test data (N_test x d)
%
% Outputs:
%   X_train_norm : Normalized training data (values in [0,1])
%   X_test_norm  : Normalized test data (values in [0,1])
%   params       : Struct with min and max computed on training data

    % Compute min and max on training data only
    x_min = min(X_train, [], 1);
    x_max = max(X_train, [], 1);
    
    % Prevent division by zero
    range = x_max - x_min;
    range(range == 0) = 1;
    
    % Normalize training data
    X_train_norm = (X_train - repmat(x_min, size(X_train, 1), 1)) ./ repmat(range, size(X_train, 1), 1);
    
    % Normalize test data using training min/max
    X_test_norm = (X_test - repmat(x_min, size(X_test, 1), 1)) ./ repmat(range, size(X_test, 1), 1);
    
    % Clip to [0, 1] (in case test data has values outside training range)
    X_test_norm = max(0, min(1, X_test_norm));
    
    % Store parameters
    params.x_min = x_min;
    params.x_max = x_max;
end