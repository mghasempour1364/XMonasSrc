function MK = eval_mk(y_true, y_pred)
% EVAL_MK: Markedness (Eq. 40)
%
% For multiclass, uses macro-averaged one-vs-rest.
%
% Inputs:
%   y_true : True labels (N x 1)
%   y_pred : Predicted labels (N x 1)
%
% Output:
%   MK : Markedness

    classes = unique(y_true);
    num_classes = length(classes);
    
    if num_classes == 2
        % Binary case
        TP = sum((y_pred == classes(1)) & (y_true == classes(1)));
        TN = sum((y_pred == classes(2)) & (y_true == classes(2)));
        FP = sum((y_pred == classes(1)) & (y_true == classes(2)));
        FN = sum((y_pred == classes(2)) & (y_true == classes(1)));
        
        PPV = TP / max(TP + FP, 1);
        NPV = TN / max(TN + FN, 1);
        MK = PPV + NPV - 1;
    else
        % Multiclass: macro-averaged one-vs-rest
        MK_per_class = zeros(num_classes, 1);
        for c = 1:num_classes
            y_true_bin = (y_true == classes(c));
            y_pred_bin = (y_pred == classes(c));
            
            TP = sum(y_pred_bin & y_true_bin);
            TN = sum(~y_pred_bin & ~y_true_bin);
            FP = sum(y_pred_bin & ~y_true_bin);
            FN = sum(~y_pred_bin & y_true_bin);
            
            PPV = TP / max(TP + FP, 1);
            NPV = TN / max(TN + FN, 1);
            MK_per_class(c) = PPV + NPV - 1;
        end
        MK = mean(MK_per_class);
    end
end