function BM = eval_bm(y_true, y_pred)
% EVAL_BM: Bookmaker Informedness (Eq. 39)
%
% For multiclass, uses macro-averaged one-vs-rest.
%
% Inputs:
%   y_true : True labels (N x 1)
%   y_pred : Predicted labels (N x 1)
%
% Output:
%   BM : Bookmaker Informedness

    classes = unique(y_true);
    num_classes = length(classes);
    
    if num_classes == 2
        % Binary case
        TP = sum((y_pred == classes(1)) & (y_true == classes(1)));
        TN = sum((y_pred == classes(2)) & (y_true == classes(2)));
        FP = sum((y_pred == classes(1)) & (y_true == classes(2)));
        FN = sum((y_pred == classes(2)) & (y_true == classes(1)));
        
        TPR = TP / max(TP + FN, 1);
        TNR = TN / max(TN + FP, 1);
        BM = abs(TPR) + abs(TNR) - 1;
    else
        % Multiclass: macro-averaged one-vs-rest
        BM_per_class = zeros(num_classes, 1);
        for c = 1:num_classes
            y_true_bin = (y_true == classes(c));
            y_pred_bin = (y_pred == classes(c));
            
            TP = sum(y_pred_bin & y_true_bin);
            TN = sum(~y_pred_bin & ~y_true_bin);
            FP = sum(y_pred_bin & ~y_true_bin);
            FN = sum(~y_pred_bin & y_true_bin);
            
            TPR = TP / max(TP + FN, 1);
            TNR = TN / max(TN + FP, 1);
            BM_per_class(c) = abs(TPR) + abs(TNR) - 1;
        end
        BM = mean(BM_per_class);
    end
end