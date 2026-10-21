function [X, y, dataset_name] = util_load_dataset(dataset_path)
% UTIL_LOAD_DATASET: Load a dataset from CSV or XLSX file
%
% This function dynamically detects the file extension and loads the data
% accordingly. It also handles the header row and separates features from labels.
%
% Inputs:
%   dataset_path : Full path to the dataset file (CSV or XLSX)
%
% Outputs:
%   X            : Feature matrix (N x d)
%   y            : Label vector (N x 1)
%   dataset_name : Name of the dataset (derived from filename without extension)

    % --- Check if file exists ---
    if ~exist(dataset_path, 'file')
        error('File not found: %s\nPlease check the path and filename.', dataset_path);
    end
    
    % --- Detect file extension ---
    [~, dataset_name, ext] = fileparts(dataset_path);
    ext = lower(ext);
    
    % --- Load data based on extension ---
    switch ext
        case '.csv'
            % Read CSV file (skip header row)
            try
                data = csvread(dataset_path, 1, 0);
            catch
                % Fallback: use readtable (if available in MATLAB R2015a+)
                % readtable was introduced in R2013b, so it should be available
                tbl = readtable(dataset_path);
                data = table2array(tbl);
                % If the first row is header, table2array will handle it
                % (it reads numeric data only, so we may need to adjust)
                if iscell(data)
                    % If data is cell array (mixed types), convert numeric columns
                    data = cell2mat(data(:, 1:end-1));
                    labels = data(:, end);
                end
            end
            
        case '.xlsx'
            % Read Excel file (skip header row)
            % xlsread returns [num, txt, raw]
            [num_data, ~, raw] = xlsread(dataset_path);
            
            % Check if num_data is empty (e.g., all data is text)
            if isempty(num_data)
                % Try to convert raw data to numeric
                % Assume first row is header, first column is index (optional)
                raw_data = raw(2:end, :); % Skip header
                % Convert to numeric matrix
                num_data = zeros(size(raw_data, 1), size(raw_data, 2));
                for i = 1:size(raw_data, 1)
                    for j = 1:size(raw_data, 2)
                        val = raw_data{i, j};
                        if ischar(val)
                            num_data(i, j) = str2double(val);
                        else
                            num_data(i, j) = val;
                        end
                    end
                end
            end
            data = num_data;
            
        otherwise
            error('Unsupported file format: %s. Please use .csv or .xlsx', ext);
    end
    
    % --- Handle potential NaN values (missing data) ---
    % (NaN values will be imputed by RMDP later)
    
    % --- Separate features and labels ---
    % Assume the last column is the label
    X = data(:, 1:end-1);
    y = data(:, end);
    
    % --- Ensure y is a column vector ---
    if size(y, 2) > 1
        y = y';
    end
    
    % --- Display dataset info ---
    fprintf('Loaded dataset: %s\n', dataset_name);
    fprintf('  Samples: %d, Features: %d, Classes: %d\n', ...
        size(X, 1), size(X, 2), length(unique(y)));
end