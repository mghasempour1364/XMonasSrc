function BA = eval_balanced_accuracy(y_true, y_pred)
% EVAL_BALANCED_ACCURACY: Balanced Accuracy (Eq. 37)
%
% For multiclass, uses macro-averaged one-vs-rest.
%
% Inputs:
%   y_true : True labels (N x 1)
%   y_pred : Predicted labels (N x 1)
%
% Output:
%   BA : Balanced Accuracy

    classes = unique(y_true);
    num_classes = length(classes);
    
    if num_classes == 2
        % Binary case
        TP = sum((y_pred == classes(1)) & (y_true == classes(1)));
        TN = sum((y_pred == classes(2)) & (y_true == classes(2)));
        P = sum(y_true == classes(1));
        N = sum(y_true == classes(2));
        BA = (TP / P + TN / N) / 2;
    else
        % Multiclass: macro-averaged one-vs-rest
        BA_per_class = zeros(num_classes, 1);
        for c = 1:num_classes
            y_true_bin = (y_true == classes(c));
            y_pred_bin = (y_pred == classes(c));
            
            TP = sum(y_pred_bin & y_true_bin);
            TN = sum(~y_pred_bin & ~y_true_bin);
            P = sum(y_true_bin);
            N = sum(~y_true_bin);
            
            sensitivity = TP / max(P, 1);
            specificity = TN / max(N, 1);
            BA_per_class(c) = (sensitivity + specificity) / 2;
        end
        BA = mean(BA_per_class);
    end
end