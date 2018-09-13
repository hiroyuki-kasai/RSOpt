function demo()
% This file is part of RSOpt package.
%
% Created by H.Kasai and B.Mishra on July 20, 2018
    
    clc; close all; clear


    %% define parameters
    N = 500;
    d = 3;    
   
    
    
    %% read dataset
    input_data = load('./dataset/psd/psd_mean_3_500_5.mat');
    A = zeros(d, d, N);
    A = input_data.x_sample{1};
    f_sol = input_data.f_sol{1}; 
    fprintf('f_sol: %.16e\n', f_sol);           

    

    %% set manifold
    problem.M = sympositivedefinitefactory_mod(d);  
    problem.ncostterms = N;     
    
    
    %% define problem
    % cost function
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
    
    
    
    %% run SRG algorithm    
    
    Init = problem.M.rand();
    
    clear options;
    options.verbosity = 1;
    options.batchsize = 10;
    options.maxepoch = 30;
    options.tolgradnorm = 1e-8;         
    options.stepsize = 0.01;       
    options.transport = 'ret_vector_locking';  
    options.maxinneriter = N;    
    [~, ~, infos_srg, options_srg] = Riemannian_srg(problem, Init, options); 
    for kk = 1 : size(infos_srg,2)
        num_grads_srg(kk) = infos_srg(kk).grad_cnt;
    end
    
      
    
    %% plots
    fs = 20;

    % Optimality gap (Train loss - optimum) versus #grads/N     
    optgap_srg = abs([infos_srg.cost] - f_sol);    

    % Optimality gap versus #grads/N
    figure;
    semilogy(num_grads_srg, optgap_srg, '-', 'LineWidth',2,'Color', [0, 0, 255]/255);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Traning loss - optimum','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SRG');
    
    % Optimality gap versus times [sec]
    figure;
    semilogy(abs([infos_srg.time]), optgap_srg, '-', 'LineWidth',2,'Color', [0, 0, 255]/255);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time [sec]','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Traning loss - optimum','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SRG');    
    
    
    % Gradient norm versus #grads/N 
    figure;
    semilogy(num_grads_srg, [infos_srg.gradnorm], '-', 'LineWidth',2,'Color', [0, 0, 255]/255);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'#grad/N','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Norm of gradient','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('R-SRG');

end

