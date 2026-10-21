function P_at_N = eval_precision_at_top_n(scores, relevant_idx, N)
% EVAL_PRECISION_AT_TOP_N: Precision@Top-N (Eq. 41)
%
% Inputs:
%   scores        : Feature importance scores (d x 1)
%   relevant_idx  : Indices of ground-truth relevant features (from domain experts)
%   N             : Number of top features to consider
%
% Output:
%   P_at_N : Precision@Top-N

    [~, sorted_idx] = sort(scores, 'descend');
    top_N_idx = sorted_idx(1:min(N, length(scores)));
    
    % Count how many of the top-N features are in the relevant set
    num_relevant = sum(ismember(top_N_idx, relevant_idx));
    
    P_at_N = num_relevant / min(N, length(scores));
end