function show_centroid_plots()
    
    clc; close all; clear

    %% Define parameters
    tolgradnorm = 1e-8;
    inner_repeat = 1;
    

if 1
    % Large instance
    maxepoch = 30;  
    N = 5000;
    d = 10;    
    cn = 5;  
    srg_varpi = 0.05;
else
    % Small instance
    maxepoch = 60;
    N = 1500;
    d = 3;    
    cn = 5; 
    srg_varpi = 0.05;
end
    

    input_filename = sprintf('./dataset/psd/psd_mean_%d_%d_%d.mat', d, N, cn);
    fprintf('Reading file %s with (d:%d N:%d cn:%d) .... ', input_filename, d, N, cn); 
    input_data = load(input_filename);        

    A = zeros(d, d, N);

    A = input_data.x_sample{1};
    fprintf('done\n');

    f_sol = input_data.f_sol{1}; 
    fprintf('f_sol: %.16e\n', f_sol);           

    
    %% Set manifold
    problem.M = sympositivedefinitefactory_mod(d);  
    problem.ncostterms = N;     

    
    
    % Cost function
    problem.cost = @cost;    
    function f = cost(X)
        f=0;
        sqrtX = sqrtm(X);
        for i=1:N
            arg = sqrtX\A(:, :, i)/sqrtX;
            if (norm(imag(eig(arg)),'fro')>1e-15)
                f = Inf;
                break;
            elseif (any(real(eig(arg))<0))
                f = Inf;
                break;
            end
            f = f + norm(logm(arg),'fro')^2;
        end
        
        f = f/(N);
    end



    % Riemannian gradient of the cost function
    problem.rgrad = @rgrad;      
    function g = rgrad(X)

        logsum = zeros(size(X,1));

        invX = pinv(X);
        for i = 1 : N
            logsum = logsum + logm(A(:, :, i) * invX);
        end

        g = 2*X*logsum;
        g = (g+g')/2;
        g = g/N;
    end
    


    % Riemannian stochastic gradient of the cost function
    problem.partialgrad = @partialgrad;
    function g = partialgrad(X, idx_batchsize)        

        m_batchsize = length(idx_batchsize); 
        
        logsum = zeros(size(X,1));
        for k = 1 : m_batchsize
            curr_index = idx_batchsize(k);
            logsum = logsum + logm(A(:, :, curr_index)\X);
        end

        g = 2*X*logsum;
        g = (g+g')/2;
        g = g/m_batchsize;
    end
        
    %     % Consistency checks
    %     checkgradient(problem)
    %     pause;
    
    
    
    %% Run algorithms    
    
    % Initialize
    Uinit = problem.M.rand();
    

    % Run R-SD
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_sd, options_sd] = steepestdescent(problem, Uinit, options); 
    
    
    % Run R-CG
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_cg, options_cg] = conjugategradient(problem, Uinit, options);     
    
    
    % Run SGD with decay step-size
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='sgd';
    options.maxepoch = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'decay';
    options.stepsize = 1e-3;      
    options.stepsize_lambda = 0.01;   
    options.transport = 'ret_vector_locking';
    options.maxinneriter = inner_repeat*N;
    [~, ~, infos_sgd, options_sgd] = Riemannian_svrg(problem, Uinit, options);

    
    % Run SVRG
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.update_type='svrg';
    options.maxepoch = maxepoch / (1 + 2 * inner_repeat);
    options.tolgradnorm = tolgradnorm;         
    options.svrg_type = 2;
    options.stepsize_type = 'fix';
    options.stepsize = 0.01;       
    options.boost = 0;    
    options.transport = 'ret_vector_locking';
    options.store_innerinfo = false; 
    options.maxinneriter = inner_repeat*N;
    [~, ~, infos_svrg, options_svrg] = Riemannian_svrg(problem, Uinit, options);


    % Run SRG
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.maxepoch = maxepoch / (1 + 2 * inner_repeat);
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'fix';
    options.stepsize = 0.01;       
    options.transport = 'ret_vector_locking';  
    options.store_innerinfo = false;
    options.maxinneriter = inner_repeat*N;    
    [~, ~, infos_srg, options_srg] = Riemannian_srg(problem, Uinit, options); 
    

    % Run SRG+
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.maxepoch = maxepoch / (1 + 2 * inner_repeat);
    options.tolgradnorm = tolgradnorm;         
    options.stepsize_type = 'fix';
    options.stepsize = 0.01;       
    options.gamma = srg_varpi;
    options.transport = 'ret_vector_locking';   
    options.store_innerinfo = false; 
    options.maxinneriter = inner_repeat*N;    
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
    

    
    
      
    
    %% Plots
    fs = 20;

    % Optimality gap (Train loss - optimum) versus #grads/N     
    optgap_sd = abs([infos_sd.cost] - f_sol);
    optgap_cg = abs([infos_cg.cost] - f_sol);
    optgap_sgd = abs([infos_sgd.cost] - f_sol);
    optgap_svrg = abs([infos_svrg.cost] - f_sol);
    optgap_srg = abs([infos_srg.cost] - f_sol);    
    optgap_srg_plus = abs([infos_srg_plus.cost] - f_sol);     
     

    % Optimality gap versus #grads/N
    figure;
    semilogy(num_grads_sd, optgap_sd,'-','LineWidth',2,'Color', [255, 128, 0]/255);                hold on;
    semilogy(num_grads_cg, optgap_cg,'-','LineWidth',2,'Color', [76, 153, 0]/255);                 hold on;    
    semilogy(num_grads_sgd, optgap_sgd,'-','LineWidth',2,'Color', [255,0,255]/255);                hold on;
    semilogy(num_grads_svrg, optgap_svrg, '-', 'LineWidth',2,'Color', [255, 0, 0]/255);            hold on;    
    semilogy(num_grads_srg, optgap_srg, '-', 'LineWidth',2,'Color', [0, 0, 255]/255);              hold on;
    semilogy(num_grads_srg_plus, optgap_srg_plus, '-.', 'LineWidth',2,'Color', [0, 0, 255]/255);   hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Traning loss - optimum','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SD', 'R-CG', 'R-SGD', 'R-SVRG', 'R-SRG', 'R-SRG+');
    
    % Optimality gap versus #grads/N
    figure;
    semilogy(abs([infos_sd.time]), optgap_sd,'-','LineWidth',2,'Color', [255, 128, 0]/255);                hold on;
    semilogy(abs([infos_cg.time]), optgap_cg,'-','LineWidth',2,'Color', [76, 153, 0]/255);                 hold on;    
    semilogy(abs([infos_sgd.time]), optgap_sgd,'-','LineWidth',2,'Color', [255,0,255]/255);                hold on;
    semilogy(abs([infos_svrg.time]), optgap_svrg, '-', 'LineWidth',2,'Color', [255, 0, 0]/255);            hold on;    
    semilogy(abs([infos_srg.time]), optgap_srg, '-', 'LineWidth',2,'Color', [0, 0, 255]/255);              hold on;
    semilogy(abs([infos_srg_plus.time]), optgap_srg_plus, '-.', 'LineWidth',2,'Color', [0, 0, 255]/255);   hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time [sec]','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Traning loss - optimum','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SD', 'R-CG', 'R-SGD', 'R-SVRG', 'R-SRG', 'R-SRG+');    
    
    
    % Gradient norm versus #grads/N 
    figure;
    semilogy(num_grads_sd, [infos_sd.gradnorm],'-','LineWidth',2,'Color', [255, 128, 0]/255);       hold on;
    semilogy(num_grads_cg, [infos_cg.gradnorm],'-','LineWidth',2,'Color', [76, 153, 0]/255);        hold on;    
    semilogy(num_grads_sgd, [infos_sgd.gradnorm],'-','LineWidth',2,'Color', [255,0,255]/255);       hold on;
    semilogy(num_grads_svrg, [infos_svrg.gradnorm], '-', 'LineWidth',2, 'Color',[255, 0, 0]/255);   hold on;
    semilogy(num_grads_srg, [infos_srg.gradnorm], '-', 'LineWidth',2,'Color', [0, 0, 255]/255);     hold on;
    semilogy(num_grads_srg_plus, [infos_srg_plus.gradnorm], '-.', 'LineWidth',2, 'Color', [0, 0, 255]/255);    hold on;     
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Norm of gradient','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SD', 'R-CG', 'R-SGD', 'R-SVRG', 'R-SRG', 'R-SRG+');
    
end

