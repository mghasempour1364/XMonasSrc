function [X_train_proc, X_test_proc, rmParams] = RMDP_main(X_train, X_test, K, tau)
% RMDP_MAIN: Robust Medical Data Preprocessing (Section 3.1)
%
% Inputs:
%   X_train : Training data (N_train x d)
%   X_test  : Test data (N_test x d)
%   K       : Number of nearest neighbors for KNN imputation
%   tau     : Z-score threshold for outlier correction
%
% Outputs:
%   X_train_proc : Preprocessed training data
%   X_test_proc  : Preprocessed test data
%   rmParams     : Struct containing all parameters fitted on training data

    % --- Step 1: KNN Imputation (fitted on training data) ---
    [X_train_imp, X_test_imp, knnParams] = RMDP_knn_impute(X_train, X_test, K);
    
    % --- Step 2: Z-score Outlier Correction (fitted on training data) ---
    [X_train_out, X_test_out, outlierParams] = RMDP_outlier_correction(X_train_imp, X_test_imp, tau);
    
    % --- Step 3: Min-Max Normalization (fitted on training data) ---
    [X_train_norm, X_test_norm, normParams] = RMDP_normalize(X_train_out, X_test_out);
    
    % --- Step 4: PCA Dimensional Harmonization (fitted on training data) ---
    [X_train_proc, X_test_proc, pcaParams] = RMDP_pca_harmonize(X_train_norm, X_test_norm);
    
    % --- Store all parameters for reproducibility ---
    rmParams.knn = knnParams;
    rmParams.outlier = outlierParams;
    rmParams.norm = normParams;
    rmParams.pca = pcaParams;
end