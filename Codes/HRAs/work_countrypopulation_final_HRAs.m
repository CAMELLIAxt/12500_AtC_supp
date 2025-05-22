%% Read data
clear;
addpath(genpath('./minFunc_2012'));
addpath(genpath('./minConf'));
data=dlmread('../../Datas/Country_pop_datas/Country_pop_all_pair.txt');
doc_diff=dlmread('../../Datas/Country_pop_datas/Country_pop_doc_info.txt');
doc_diff=-doc_diff(:,1);

n_anno=max(data(:,1));
n_obj=max(max(data(:,2:3)));

pair=cell(n_anno,1);
rng('shuffle');
keep_ratio = 1.00;
out_f = fopen('./cp16/temp2.txt', 'a+');
for i=1:n_anno
    dp=data(data(:,1)==i, 2:3);
    [mlen ,~] = size(dp);
    dp=dp(1:ceil(mlen*keep_ratio), :);
    
    pair{i}=dp;
end

exps=20;
res = cell(exps, 1);
legends_cell = cell(exps, 1);
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h','+','o','*','.','x','s','d','^','v','>','<','p','h'};
res_idx = 1;


%% set up initial parametmers 
sc=1;
para=struct('reg_0', 10., 'reg_s', 0, 'reg_alpha', 0,  'maxiter', 600, 's0', 0,...
             'uni_weight', true, 'verbose', true, 'tol', 1e-5);
para.lr=5*10e-4;
para.alpha_rate = 0.25;

%% Ones + HRA-G
name='Ones + HRA-G s->a';
fprintf([name '\n']);
para.algo='HRA-G';
para.opt_method='s->a+GD';
alpha_init=ones(n_anno,1);
[s,alpha, obj, iter]=alter(ones(n_obj, 1)*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx} = {name, auc, kendall, s}; 
res_idx=res_idx+1;

%% Ones + HRA-N
name='Ones + HRA-N s->a';
fprintf([name '\n']);
para.algo='HRA-N';
para.opt_method='s->a+GD';
[s,alpha, obj, iter]=alter(ones(n_obj, 1)*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx} = {name, auc, kendall, s}; 
res_idx=res_idx+1;

%% Ones + HRA-E
name='Ones + HRA-E s->a';
fprintf([name '\n']);
para.algo='HRA-E';
para.opt_method='s->a+GD';
[s,alpha, obj, iter]=alter(ones(n_obj, 1)*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx} = {name, auc, kendall, s}; 
res_idx=res_idx+1;

%% Final clean up
res=res(1:res_idx-1);

for i = 1:res_idx-1
    name = res{i}{1};
    s = res{i}{4};
    filename = sprintf('Datas/Country_pop_datas/Country_pop_scores_%s.txt', strrep(name, ' ', '_'));
    dlmwrite(filename, s, 'delimiter', '\t', 'precision', '%.6f');
    fprintf('Saved scores for %s to %s\n', name, filename);
end