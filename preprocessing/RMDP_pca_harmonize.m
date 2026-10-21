function [X_train_pca, X_test_pca, params] = RMDP_pca_harmonize(X_train, X_test, var_threshold)
% RMDP_PCA_HARMONIZE: PCA-based dimensional harmonization (Eq. 5)
%
% Inputs:
%   X_train       : Training data (N_train x d)
%   X_test        : Test data (N_test x d)
%   var_threshold : Variance threshold for selecting principal components
%                   (default: 0.95, i.e., retain 95% of variance)
%
% Outputs:
%   X_train_pca : PCA-transformed training data
%   X_test_pca  : PCA-transformed test data
%   params      : Struct with PCA transformation matrix and mean

    if nargin < 3
        var_threshold = 0.95;
    end
    
    % Compute mean on training data
    mu = mean(X_train, 1);
    
    % Center training data
    X_train_centered = X_train - repmat(mu, size(X_train, 1), 1);
    
    % Compute covariance matrix (on training data only)
    cov_matrix = cov(X_train_centered);
    
    % Compute eigenvectors and eigenvalues
    [eigenvectors, eigenvalues] = eig(cov_matrix);
    eigenvalues = diag(eigenvalues);
    
    % Sort eigenvalues in descending order
    [sorted_eigenvalues, sort_idx] = sort(eigenvalues, 'descend');
    sorted_eigenvectors = eigenvectors(:, sort_idx);
    
    % Determine number of components to retain
    cumulative_var = cumsum(sorted_eigenvalues) / sum(sorted_eigenvalues);
    num_components = find(cumulative_var >= var_threshold, 1);
    if isempty(num_components)
        num_components = size(X_train, 2);
    end
    
    % Select top eigenvectors
    U = sorted_eigenvectors(:, 1:num_components);
    
    % Transform training data
    X_train_pca = X_train_centered * U;
    
    % Transform test data using training mean and eigenvectors
    X_test_centered = X_test - repmat(mu, size(X_test, 1), 1);
    X_test_pca = X_test_centered * U;
    
    % Store parameters
    params.mu = mu;
    params.U = U;
    params.num_components = num_components;
    params.explained_variance = cumulative_var(num_components);
end