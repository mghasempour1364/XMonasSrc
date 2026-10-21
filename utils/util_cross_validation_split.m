function [train_idx, test_idx] = util_cross_validation_split(y, k, fold)
% UTIL_CROSS_VALIDATION_SPLIT: Stratified K-Fold Cross-Validation Split
%
% Inputs:
%   y    : Label vector (N x 1)
%   k    : Number of folds
%   fold : Current fold index (1 to k)
%
% Outputs:
%   train_idx : Indices of training samples
%   test_idx  : Indices of test samples

    classes = unique(y);
    train_idx = [];
    test_idx = [];
    
    for c = 1:length(classes)
        class_idx = find(y == classes(c));
        class_idx = class_idx(randperm(length(class_idx)));
        
        n_class = length(class_idx);
        fold_size = floor(n_class / k);
        
        if fold < k
            test_start = (fold - 1) * fold_size + 1;
            test_end = fold * fold_size;
        else
            test_start = (fold - 1) * fold_size + 1;
            test_end = n_class;
        end
        
        test_idx = [test_idx; class_idx(test_start:test_end)];
        train_idx = [train_idx; class_idx(setdiff(1:n_class, test_start:test_end))];
    end
    
    % Shuffle training indices
    train_idx = train_idx(randperm(length(train_idx)));
end