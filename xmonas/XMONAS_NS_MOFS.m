function [S_NS, model_NN] = XMONAS_NS_MOFS(X_train, y_train, X_test, y_test, symbolic_rules, alpha, beta)
% XMONAS_NS_MOFS: Neuro-Symbolic Feature Scoring (Section 3.2)

    [N_train, d] = size(X_train);
    
    % --- Step 1: Train a lightweight neural network ---
    hidden_layers = [16, 8]; % Smaller network for speed
    net = feedforwardnet(hidden_layers);
    net.trainParam.epochs = 50;
    net.trainParam.lr = 0.01;
    net.trainParam.showWindow = false;
    net = train(net, X_train', y_train');
    model_NN = net;
    
    % --- Step 2: Compute neural relevance scores ---
    % Use permutation importance (more robust than gradient)
    f_j = zeros(d, 1);
    base_pred = net(X_train');
    base_loss = -mean(sum(y_train' .* log(base_pred + 1e-10), 1));
    
    for j = 1:d
        X_perm = X_train;
        X_perm(:, j) = X_perm(randperm(N_train), j); % Permute feature j
        perm_pred = net(X_perm');
        perm_loss = -mean(sum(y_train' .* log(perm_pred + 1e-10), 1));
        f_j(j) = abs(perm_loss - base_loss); % Importance = loss increase
    end
    
    % Normalize neural relevance to [0, 1]
    if max(f_j) > 0
        f_j = f_j / max(f_j);
    else
        f_j = ones(d, 1) / d;
    end
    
    % --- Step 3: Compute symbolic satisfaction scores ---
    if isempty(symbolic_rules)
        % If no rules, use uniform scores (all features equally plausible)
        C_k = ones(d, 1);
    else
        C_k = zeros(d, 1);
        for j = 1:d
            sat_count = 0;
            for k = 1:length(symbolic_rules)
                if symbolic_rules{k}(j)
                    sat_count = sat_count + 1;
                end
            end
            C_k(j) = sat_count / max(length(symbolic_rules), 1);
        end
    end
    
    % --- Step 4: Compute combined score (Eq. 12) ---
    S_NS = alpha * f_j + beta * C_k;
    
    % Ensure scores are in [0, 1]
    S_NS = max(0, min(1, S_NS));
end