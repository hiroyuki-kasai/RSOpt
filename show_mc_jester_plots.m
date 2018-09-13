function  show_mc_jester_plots()
% This file is part of RSOpt package.
%
% Created by H.Kasai and B.Mishra on Sep. 13, 2018

    clc; close all; clear


    %% set parameters
    maxepoch = 36;
    N = 100;
    d = 24983;   
    r = 5;
    tolgradnorm = 1e-8;
    batchsize = 1;
    inner_repeat = 1;
    srg_varpi = 0.5;
        
    
    
    %% generate dataset
    [samples, samples_test, samples_valid, values, indicator, values_test, indicator_test, data_ls, data_test, ~] = generate_mc_data_jester();


    
    %% set manifold
    problem.M = grassmannfactory(d, r);
    problem.ncostterms = N;

    
    
    %% define problem
    % cost function
    problem.cost = @mc_cost;
    function f = mc_cost(U)
        W = mylsqfit(U, samples);
        f = 0.5*norm(indicator.*(U*W') - values, 'fro')^2;
        f = f/N;    
    end

    % Euclidean gradient of the cost function
    problem.egrad = @mc_egrad;
    function g = mc_egrad(U)
        W = mylsqfit(U, samples);
        g = (indicator.*(U*W') - values)*W;
        g = g/N;
    end

    % Euclidean stochastic gradient of the cost function
    problem.partialegrad = @mc_partialegrad;
    function g = mc_partialegrad(U, idx_batchsize)
        g = zeros(d, r);
        m_batchsize = length(idx_batchsize);
        for ii = 1 : m_batchsize
            colnum = idx_batchsize(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;

    end

    function W = mylsqfit(U, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);

            % Solve a simple least squares problem to populate U
            W(ii,:) = (U_Omega\values_Omega)';
        end
    end


    %     % Consistency checks
    %     checkgradient(problem)
    %     pause;
    

    
    
    %% run algorithms
    
    % Initialize
    Uinit = problem.M.rand();     
    
    
    % R-SD
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_sd, options_sd] = steepestdescent(problem, Uinit, options);    

    
    % R-CG
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_cg, options_cg] = conjugategradient(problem, Uinit, options);        

    
    % R-SGD with decay step-size
    clear options;
    options.verbosity = 1;
    options.batchsize = batchsize;
    options.update_type='sgd';
    options.maxepoch = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'decay';
    options.stepsize = 1e-7;                 
    options.stepsize_lambda = 1e-3;  
    options.transport = 'ret_vector'; 
    options.maxinneriter = N;
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_sgd, options_sgd] = Riemannian_svrg(problem, Uinit, options);     

    
    % R-SVRG
    clear options;
    options.verbosity = 1;
    options.batchsize = batchsize;
    options.update_type='svrg';
    options.maxepoch = maxepoch / (1 + 2 * inner_repeat);
    options.tolgradnorm = tolgradnorm;         
    options.svrg_type = 1;
    options.stepsize_type = 'fix';
    options.stepsize = 1e-6; 
    options.boost = 0; 
    options.svrg_type = 1; % effective only for R-SVRG variants
    options.transport = 'ret_vector';    
    options.maxinneriter = inner_repeat * N;                  
    options.statsfun = @mc_mystatsfun;
    [~, ~, infos_svrg, options_svrg] = Riemannian_svrg(problem, Uinit, options);   

    
    % R-SRG
    clear options;
    options.verbosity = 1;
    options.batchsize = batchsize;
    options.update_type='srg';
    options.maxepoch = maxepoch / (1 + 2 * inner_repeat);
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'fix';
    options.stepsize = 1e-6; 
    options.transport = 'ret_vector';    
    options.maxinneriter = inner_repeat * N;  
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_srg, options_srg] = Riemannian_srg(problem, Uinit, options); 
    
    
    % R-SRG+
    clear options;
    options.verbosity = 1;
    options.batchsize = batchsize;
    options.update_type='srg';
    options.maxepoch = maxepoch / (1 + 2 * inner_repeat);
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'fix';
    options.stepsize = 1e-6; 
    options.transport = 'ret_vector';    
    options.maxinneriter = inner_repeat * N; 
    options.gamma = srg_varpi;
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_srg_plus, options_srg_plus] = Riemannian_srg(problem, Uinit, options);      

    
    % Calculate # of gradient evaluations
    num_grads_sd = (1:length([infos_sd.cost])) - 1; % N*options_sd.maxiter;
    num_grads_cg = (1:length([infos_cg.cost])) - 1; % N*options_sd.maxiter;    
    num_grads_sgd = ceil(options_sgd.maxinneriter/N)*((1:length([infos_sgd.cost])) - 1); % options.maxepoch*(options_sgd.maxinneriter);
    num_grads_svrg = ceil((N + 2*options_svrg.maxinneriter)/N)*((1:length([infos_svrg.cost])) - 1); %options.maxepoch*(N + options_svrg.maxinneriter); % Per epoch we compute equivalent of 2 batch grads.
    for kk = 1 : size(infos_srg,2)
        num_grads_srg(kk) = infos_srg(kk).grad_cnt;
    end
    num_grads_srg = num_grads_srg/N; 
    for kk = 1 : size(infos_srg_plus,2)
        num_grads_srg_plus(kk) = infos_srg_plus(kk).grad_cnt;
    end
    num_grads_srg_plus = num_grads_srg_plus/N; 
    

    
    

    %% plots
    fs = 20;

    % Train MSE versus #grads/N    
    figure;
    semilogy(num_grads_sd, [infos_sd.cost] * 2 * N / data_ls.nentries,'-','LineWidth',2,'Color', [255, 128, 0]/255);        hold on;
    semilogy(num_grads_cg, [infos_cg.cost] * 2 * N / data_ls.nentries,'-','LineWidth',2,'Color', [76, 153, 0]/255);         hold on;
    semilogy(num_grads_sgd, [infos_sgd.cost]  * 2 * N / data_ls.nentries,'-','LineWidth',2,'Color', [255,0,255]/255);       hold on;    
    semilogy(num_grads_svrg, [infos_svrg.cost] * 2 * N / data_ls.nentries, '-', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    semilogy(num_grads_srg, [infos_srg.cost] * 2 * N / data_ls.nentries, '-', 'LineWidth',2,'Color', [0, 0, 255]/255);      hold on;
    semilogy(num_grads_srg_plus, [infos_srg_plus.cost] * 2 * N / data_ls.nentries, '-.', 'LineWidth',2,'Color', [0, 0, 255]/255);         hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Means square error on train set \Gamma','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SD', 'R-CG', 'R-SGD', 'R-SVRG', 'R-SRG', 'R-SRG+'); 
    
    % Train MSE versus #grads/N    
    fs = 20;
    figure;
    semilogy(num_grads_sd, [infos_sd.cost_test] * 2 * N / data_test.nentries,'-','LineWidth',2,'Color', [255, 128, 0]/255);     hold on;
    semilogy(num_grads_cg, [infos_cg.cost_test] * 2 * N / data_test.nentries,'-','LineWidth',2,'Color', [76, 153, 0]/255);      hold on;
    semilogy(num_grads_sgd, [infos_sgd.cost_test]  * 2 * N / data_test.nentries,'-','LineWidth',2,'Color', [255,0,255]/255);    hold on;    
    semilogy(num_grads_svrg, [infos_svrg.cost_test] * 2 * N / data_test.nentries, '-', 'LineWidth',2,'Color', [255, 0, 0]/255); hold on;
    semilogy(num_grads_srg, [infos_srg.cost_test] * 2 * N / data_test.nentries, '-', 'LineWidth',2,'Color', [0, 0, 255]/255);   hold on;
    semilogy(num_grads_srg_plus, [infos_srg_plus.cost_test] * 2 * N / data_test.nentries, '-.', 'LineWidth',2,'Color', [0, 0, 255]/255);         hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Means square error on test set \Phi','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SD', 'R-CG', 'R-SGD', 'R-SVRG', 'R-SRG', 'R-SRG+');     
    
    
    % Train MSE versus #grads/N    
    fs = 20;
    figure;
    semilogy([infos_sd.time], [infos_sd.cost_test] * 2 * N / data_test.nentries,'-','LineWidth',2,'Color', [255, 128, 0]/255);      hold on;
    semilogy([infos_cg.time], [infos_cg.cost_test] * 2 * N / data_test.nentries,'-','LineWidth',2,'Color', [76, 153, 0]/255);       hold on;
    semilogy([infos_sgd.time], [infos_sgd.cost_test]  * 2 * N / data_test.nentries,'-','LineWidth',2,'Color', [255,0,255]/255);     hold on;    
    semilogy([infos_svrg.time], [infos_svrg.cost_test] * 2 * N / data_test.nentries, '-', 'LineWidth',2,'Color', [255, 0, 0]/255); hold on;
    semilogy([infos_srg.time], [infos_srg.cost_test] * 2 * N / data_test.nentries, '-', 'LineWidth',2,'Color', [0, 0, 255]/255);   hold on;
    semilogy([infos_srg_plus.time], [infos_srg_plus.cost_test] * 2 * N / data_test.nentries, '-.', 'LineWidth',2,'Color', [0, 0, 255]/255);         hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time [sec]','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Means square error on test set \Phi','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SD', 'R-CG', 'R-SGD', 'R-SVRG', 'R-SRG', 'R-SRG+');       
    

    function [samples, samples_test, samples_valid, values, indicator, values_test, indicator_test, data_ls, data_test, data_valid] = generate_mc_data_jester()

        load('./dataset/jester/jester_mat.mat'); % Original size [24983,100]
        
        
        samples_valid = [];
        data_valid = [];


        %% ------ [start] BM code from Scaled SGD -----
        %% Randomly select nu rows and creat the data structure
        nu = d; % Number of users selected


        p = randperm(size(A,1), nu);
        A = A(p, :); % Matrix of size nu-by-100
        Avec = A(:);

        Avecindices = 1:length(Avec);
        Avecindices = Avecindices';
        i = ones(length(Avec),1);
        i(Avec == 99) = 0;
        Avecindices_final = Avecindices(logical(i));
        [I, J] = ind2sub([size(A,1)  100],Avecindices_final);

        Avecsfinall = Avec(logical(i));


        [Isort, indI] = sort(I,'ascend');


        data_real.rows = Isort;
        data_real.cols = J(indI);
        data_real.entries = Avecsfinall(indI);
        data_real.nentries = length(data_real.entries);

        % Test data: two ratings per user
        [~,IA,~] = unique(Isort,'stable');
        data_ts_ind = [];
        for ii = 1 : length(IA)
            if ii < length(IA)
                inneridx = randperm(IA(ii+1) - IA(ii), 2);
            else
                inneridx = randperm(length(data_real.entries) +1 - IA(ii), 2);
            end
           data_ts_ind = [data_ts_ind; IA(ii) + inneridx' - 1];
        end


        data_test.rows = data_real.rows(data_ts_ind);
        data_test.cols = data_real.cols(data_ts_ind);
        data_test.entries = data_real.entries(data_ts_ind);
        data_test.nentries = length(data_test.rows);


        % Train data
        data_ls = data_real;
        data_ls.rows(data_ts_ind) = [];
        data_ls.cols(data_ts_ind) = [];
        data_ls.entries(data_ts_ind) = [];
        data_ls.nentries = length(data_ls.rows);


        % Permute train data
        random_order = randperm(length(data_ls.rows));
        data_ls.rows = data_ls.rows(random_order);
        data_ls.cols = data_ls.cols(random_order);
        data_ls.entries = data_ls.entries(random_order);
        

        % Dimensions and options
        n = size(A, 1);
        m = size(A, 2);

        %% ------ [end] BM code from Scaled SGD -----



        %% for train set
        values = sparse(data_ls.rows, data_ls.cols, data_ls.entries, n, m);
        indicator = sparse(data_ls.rows, data_ls.cols, 1, n, m);

        % Creat the cells
        samples(m).colnumber = []; % Preallocate memory.
        for k = 1 : m
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator(:, k)); % find known row indices
            values_col = values(idx, k); % the non-zero entries of the column

            samples(k).indicator = idx;
            samples(k).values = values_col;
            samples(k).colnumber = k;
        end


        %% for test set
        values_test = sparse(data_test.rows, data_test.cols, data_test.entries, n, m);
        indicator_test = sparse(data_test.rows, data_test.cols, 1, n, m);    

        samples_test(m).colnumber = [];
        for k = 1 : m
            % Pull out the relevant indices and revealed entries for this column
            idx = find(indicator_test(:, k)); % find known row indices
            values_col = values_test(idx, k); % the non-zero entries of the column

            samples_test(k).indicator = idx;
            samples_test(k).values = values_col;
            samples_test(k).colnumber = k;
        end 

        d = n;
        N = m;
    end

 
    function stats = mc_mystatsfun(problem, U, stats)
        global indicator_valid;
        global values_valid;     

        W = mylsqfit(U, samples_test);
        f_test = 0.5*norm(indicator_test.*(U*W') - values_test, 'fro')^2;
        f_test = f_test/N;
        stats.cost_test = f_test;

        if ~isempty(samples_valid) 
            W = mylsqfit(U, samples_valid);
            f_valid = 0.5*norm(indicator_valid.*(U*W') - values_valid, 'fro')^2;
            f_valid = f_valid/N;
            stats.cost_valid = f_valid;   
        end

    end    
    
end