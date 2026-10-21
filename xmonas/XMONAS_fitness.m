function F = XMONAS_fitness(subset, X_train, y_train, X_val, y_val, S_NS)
% XMONAS_FITNESS: Multi-objective fitness function (Eq. 20)
%
% Inputs:
%   subset   : Binary vector indicating selected features
%   X_train  : Training data
%   y_train  : Training labels
%   X_val    : Validation data
%   y_val    : Validation labels
%   S_NS     : Neuro-symbolic scores
%
% Output:
%   F : Fitness vector [F1, F2, F3]
%       F1: Classification accuracy (maximize)
%       F2: Feature compactness (minimize)
%       F3: Symbolic penalty (minimize)

    % --- Initialize F (to avoid "Output argument not assigned" error) ---
    F = [0, size(X_train, 2), 0]; % Default: 0 accuracy, all features, no penalty
    
    % --- Check if subset is empty ---
    selected_idx = find(subset);
    if isempty(selected_idx)
        % If no features selected, return default fitness
        F = [0, size(X_train, 2), 0];
        return;
    end
    
    % --- F1: Classification accuracy ---
    try
        X_train_sel = X_train(:, selected_idx);
        X_val_sel = X_val(:, selected_idx);
        
        % Use k-NN for simplicity (available in MATLAB R2015a)
        mdl = fitcknn(X_train_sel, y_train, 'NumNeighbors', 5);
        y_pred = predict(mdl, X_val_sel);
        F1 = mean(y_pred == y_val);
    catch
        % If training fails, set accuracy to 0
        F1 = 0;
    end
    
    % --- F2: Feature compactness (number of selected features) ---
    F2 = sum(subset);
    
    % --- F3: Symbolic penalty (placeholder) ---
    % This can be computed based on symbolic rules
    F3 = 0;
    
    % --- Return fitness vector ---
    F = [F1, F2, F3];
end