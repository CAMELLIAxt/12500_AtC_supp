%% ========================================================================
%  SETUP
%  ========================================================================
clear;

addpath(genpath('./minFunc_2012')); 
addpath(genpath('./minConf')); 

data_sources = {'M1', 'M2'};
pair_files = {'all_pair_M1.txt', 'all_pair_M2.txt'};


algorithms_special = {'BTL', 'TCV'};
algorithms_alter = {'HRA-G', 'HRA-N', 'HRA-E', 'CrowdBT', 'CrowdTCV'};


para_hra = struct('reg_0', 10.0, 'reg_s', 1.1, 'reg_alpha', 1.1,  'maxiter', 10, 's0', 0, 'uni_weight', true, 'verbose', false, 'tol', 1e-5);
para_hra.opt_method = 's->a+GD'; % or 'a->s+GD'
para_hra.lr = 1e-3;

% Crowd
para_crowd = struct('reg_0', 1.0, 'reg_s', 1.0, 'reg_alpha', 1.0,  'maxiter', 10, 's0', 0, 'uni_weight', true, 'verbose', false, 'tol', 1e-5);
para_crowd.opt_method = 's->a+newton+crowdbt';


%% ========================================================================
%  MAIN LOOP
%  ========================================================================

for d_idx = 1:length(data_sources)
    source_name = data_sources{d_idx};
    pair_file = pair_files{d_idx};
    
    fprintf('\n\n======================================================\n');
    fprintf('Processing Data Source: %s (from %s)\n', source_name, pair_file);
    fprintf('======================================================\n\n');
    
    data = dlmread(pair_file);
    
    n_anno = max(data(:,1));
    n_obj = max(max(data(:,2:3)));
    
    pair = cell(n_anno,1);
    for i = 1:n_anno
        pair{i} = data(data(:,1)==i, 2:3);
    end
    
    
    % --- (BTL, TCV) ---
    for i = 1:length(algorithms_special)
        algo_name = algorithms_special{i};
        fprintf('Running algorithm: %s\n', algo_name);
        
        if strcmp(algo_name, 'BTL')
            para = para_hra; 
            para.algo = 'HRA-G';
        elseif strcmp(algo_name, 'TCV')
            para = para_crowd; 
            para.algo = 'CrowdTCV'; 
        end
        
        s_init = randn(n_obj,1); 
        alpha_init = ones(n_anno,1); 
        opt_s = struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300);
        
        scores = minFunc(@func_s, s_init, opt_s, alpha_init, para, pair);
        
        filename = sprintf('score_%s_%s.txt', source_name, strrep(algo_name, ' ', '_'));
        dlmwrite(filename, scores, 'delimiter', '\t', 'precision', '%.6f');
        fprintf('Saved scores for %s to %s\n', algo_name, filename);
    end
    
    for i = 1:length(algorithms_alter)
        algo_name = algorithms_alter{i};
        fprintf('Running algorithm: %s\n', algo_name);
        
        if startsWith(algo_name, 'HRA')
            para = para_hra;
        else % CrowdBT, CrowdTCV
            para = para_crowd;
        end
        para.algo = algo_name;
        
        s_init = ones(n_obj, 1); 
        alpha_init = ones(n_anno, 1);
        
        [scores, alpha, obj, iter] = alter(s_init, alpha_init, pair, para);
        
        filename = sprintf('score_%s_%s.txt', source_name, strrep(algo_name, ' ', '_'));
        dlmwrite(filename, scores, 'delimiter', '\t', 'precision', '%.6f');
        fprintf('Saved scores for %s to %s\n', algo_name, filename);
    end
end

fprintf('\n\nAll processing finished.\n');
