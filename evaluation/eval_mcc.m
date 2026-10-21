function MCC = eval_mcc(y_true, y_pred)
% EVAL_MCC: Matthews Correlation Coefficient (Eq. 38)
%
% For multiclass, uses macro-averaged one-vs-rest.
%
% Inputs:
%   y_true : True labels (N x 1)
%   y_pred : Predicted labels (N x 1)
%
% Output:
%   MCC : Matthews Correlation Coefficient

    classes = unique(y_true);
    num_classes = length(classes);
    
    if num_classes == 2
        % Binary case
        TP = sum((y_pred == classes(1)) & (y_true == classes(1)));
        TN = sum((y_pred == classes(2)) & (y_true == classes(2)));
        FP = sum((y_pred == classes(1)) & (y_true == classes(2)));
        FN = sum((y_pred == classes(2)) & (y_true == classes(1)));
        
        numerator = TP * TN - FP * FN;
        denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
        if denominator == 0
            MCC = 0;
        else
            MCC = numerator / denominator;
        end
    else
        % Multiclass: macro-averaged one-vs-rest
        MCC_per_class = zeros(num_classes, 1);
        for c = 1:num_classes
            y_true_bin = (y_true == classes(c));
            y_pred_bin = (y_pred == classes(c));
            
            TP = sum(y_pred_bin & y_true_bin);
            TN = sum(~y_pred_bin & ~y_true_bin);
            FP = sum(y_pred_bin & ~y_true_bin);
            FN = sum(~y_pred_bin & y_true_bin);
            
            numerator = TP * TN - FP * FN;
            denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
            if denominator == 0
                MCC_per_class(c) = 0;
            else
                MCC_per_class(c) = numerator / denominator;
            end
        end
        MCC = mean(MCC_per_class);
    end
end