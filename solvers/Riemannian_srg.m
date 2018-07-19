function [x, cost, info, options] = Riemannian_srg(problem, x, options)
% The Riemannian stochastic recursive gradient (SRG) algorithms.
%
% function [x, cost, info, options] = Riemannian_srg(problem)
% function [x, cost, info, options] = Riemannian_srg(problem, x)
% function [x, cost, info, options] = Riemannian_srg(problem, x, options)
% function [x, cost, info, options] = Riemannian_srg(problem, [], options)
%
% Apply the Riemannian SRG algorithm to the problem defined
% in the problem structure, starting at x if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x as [] (the empty matrix).
%
% The solver mimics other solvers of Manopt with two additonal input
% requirements: problem.ncostterms and problem.partialegrad.
%
% problem.ncostterms has the number of samples, e.g., N samples.
%
% problem.partialegrad takes input a current point of the manifold and
% index of batchsize.
%
% Some of the options of the solver are specifict to this file. Please have
% a look below.
%
% The solver is based on the paper by
% H. Kasai, H. Sato, and B. Mishra,
% "Riemannian stochastic recursive gradient," ICML2018, 2018.



% Original authors: Hiroyuki Kasai <kasai@is.uec.ac.jp>, 
%                   Bamdev Mishra <bamdevm@gmail.com>, and
%                   Hiroyuki Sato <hsato@amp.i.kyoto-u.ac.jp>, Feb., 2018.
  
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
            ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', ...
            'No partial gradient provided. The algorithm will likely abort.');
    end
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    
    
    % Total number of samples
    N = problem.ncostterms;
    
    % Set local defaults
    localdefaults.maxepoch = 100;  % Maximum number of epochs.
    localdefaults.maxinneriter = N;  % Maximum number of sampling per epoch.
    localdefaults.stepsize = 0.1;  % Initial stepsize guess.
    localdefaults.stepsize_type = 'fix'; % Stepsize type. Other possibilities are 'fix' and 'decay'.
    localdefaults.stepsize_lambda = 0.1; % lambda is a weighting factor while using stepsize_typ='decay'.
    localdefaults.tolgradnorm = 1.0e-6; % Batch grad norm tolerance.
    localdefaults.batchsize = 1;  % Batchsize.
    localdefaults.verbosity = 0;  % Output verbosity. Other localdefaults are 1 and 2.
    localdefaults.store_innerinfo = false; % Store information at each update. High memory requirements. Only to be used for debugging.
    localdefaults.gamma = 0; % Threshold parameter for R-SRG+.
    localdefaults.transport = 'ret_vector';
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    
    stepsize0 = options.stepsize;
    batchsize = options.batchsize;
    
    
    % Total number of batches
    totalbatches = ceil(options.maxinneriter/batchsize);
    
    
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);
    
    % Save stats in a struct array info, and preallocate.
    epoch = 0;
    grad_cnt = 0;
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxepoch+1)).epoch = [];
    info(min(10000, options.maxepoch+1)).cost = [];
    info(min(10000, options.maxepoch+1)).time = [];
    info(min(10000, options.maxepoch+1)).gradnorm = [];
    
    % Initialize innerinfo
    iter = 0;
    inneriter = 0;
    if options.store_innerinfo
        innerstats = saveinnerstats();
        innerinfo(1) = innerstats;
        info(1).innerinfo = innerinfo;
        innerinfo(min(10000, totalbatches+1)).inneriter = []; % HK
    end
    
    if options.gamma > 0
        mode = 'R-SRG(+)';
    else
        mode = 'R-SRG';
    end
    
    
    if options.verbosity > 0
        fprintf('\n-------------------------------------------------------\n');
        fprintf('%s:  epoch\t               cost val\t    grad. norm\t stepsize\n', mode);
        fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, 0, cost, gradnorm,stepsize0);
        
        
        if options.verbosity > 1
            fprintf('\n             inneriter\t               cost val\t    grad. norm\n');
        end
    end
    
    % store initial full grad
    grad0 = grad;

    
    % Main loop over epoch.
    for epoch = 1 : options.maxepoch
        
        % Draw the samples with replacement.
        perm_idx = randi(N, 1, options.maxinneriter);
        
        
        % Update stepsize
        if strcmp(options.stepsize_type, 'decay')
            stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).

        elseif strcmp(options.stepsize_type, 'fix')
            stepsize = stepsize0; % Fixed stepsize.

        else
            error(['Unknown options.stepsize_type. ' ...
                'Should be fix or decay.']);
        end
        
        
        % Update x with full gradient
        x_prev = x;
        if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')           
            x =  problem.M.exp(x, grad0, -stepsize);
        elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')                
            x =  problem.M.retr(x, grad0, -stepsize);
        else
            x =  problem.M.retr(x, grad0, -stepsize);
        end
        newkey = storedb.getNewKey(); 

        
        % Store 
        v_prev = grad0;
        move   = - grad0 * stepsize; 
        
        
        if options.gamma > 0
            grad0_norm = problem.M.norm(x_prev, v_prev);
        end 
        
        
        % Increment grad_cnt for full grad
        grad_cnt = grad_cnt + N;
        
        
        elapsed_time = 0;
        break_inneriter = 0;
        % Per epoch: main loop over samples.
        for inneriter = 1 : totalbatches
            
            % Set start time
            start_time = tic;
            
            % Pick a sample of size batchsize
            start_index = (inneriter - 1)* batchsize + 1;
            end_index = min(inneriter * batchsize, options.maxinneriter);
            idx_batchsize = perm_idx(start_index : end_index);
            
            % Compute the gradient on this batch.
            partialgrad = getPartialGradient(problem, x, idx_batchsize, storedb, key);
            partialgrad_prev = getPartialGradient(problem, x_prev, idx_batchsize);            
            
          
            
            % Update stepsize
            if strcmp(options.stepsize_type, 'decay')
                stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
                
            elseif strcmp(options.stepsize_type, 'fix')
                stepsize = stepsize0; % Fixed stepsize.
                
            else
                error(['Unknown options.stepsize_type. ' ...
                    'Should be fix or decay.']);
            end
            
            
            % Update partialgrad
            if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')
                % parallel translation
                
                v_trans = problem.M.paratransp(x_prev, move, v_prev);  
                partialgrad_prev_trans = problem.M.paratransp(x_prev, move, partialgrad_prev);  
                
            elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')

                v_trans = problem.M.transp_locking(x_prev, move, x, v_prev); 
                partialgrad_prev_trans = problem.M.transp_locking(x_prev, move, x, partialgrad_prev);                  

            else
                % Vector transport
                
                v_trans = problem.M.transp(x_prev, x, v_prev); 
                partialgrad_prev_trans = problem.M.transp(x_prev, x, partialgrad_prev);
            end

            % Update partialgrad to reduce variance by
            % taking a linear combination with old gradients.
            % We make the combination
            % partialgrad + partialgrad_prev_trans - v_trans.
            partialgrad = problem.M.lincomb(x, 1, v_trans, 1, partialgrad);
            partialgrad = problem.M.lincomb(x, 1, partialgrad, -1, partialgrad_prev_trans);
                
                

            % Update x
            if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')           
                xnew =  problem.M.exp(x, partialgrad, -stepsize);
            else
                xnew =  problem.M.retr(x, partialgrad, -stepsize);
            end
            newkey = storedb.getNewKey();
            
            
            % Elapsed time
            elapsed_time = elapsed_time + toc(start_time);
            
            iter = iter + 1; % Total number updates.
            
            % store inner infos
            if options.store_innerinfo
                newcost = problem.cost(xnew);
                partialgradnorm = problem.M.norm(xnew, partialgrad);
                cost = newcost;
                key = newkey;
                
                innerstats = saveinnerstats();
                innerinfo(inneriter) = innerstats;
                if options.verbosity > 1
                    fprintf('%s: %5d (%5d)\t%+.16e\t%.8e\t%.8e\n', mode, inneriter, epoch, cost, partialgradnorm, stepsize);
                end
            end
            
            
            % store variable
            x_prev  = x;  
            v_prev  = partialgrad;
            move    = - partialgrad * stepsize;
            
            x = xnew;
            
            % Increment grad_cnt for stochstic grad
            grad_cnt = grad_cnt + batchsize;
            
            if options.gamma > 0
                v_prev_norm = problem.M.norm(x_prev, v_prev);

                if (v_prev_norm <  options.gamma*grad0_norm) && inneriter > 1
                    break_inneriter = inneriter;
                    break;
                end
            end            
        end
        
        
        elapsed_time = elapsed_time + toc(tic);


        % Calculate cost, grad, and gradnorm
        [newcost, newgrad] = getCostGrad(problem, xnew, storedb, newkey);
        newgradnorm = problem.M.norm(xnew, newgrad);
        grad0 = newgrad;
        
        % Transfer iterate info
        %x = xnew;
        cost = newcost;
        key = newkey;
        gradnorm = newgradnorm;
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        
        if options.store_innerinfo
            stats.innerinfo = innerinfo;
        end
        info(epoch+1)= stats;
        if options.store_innerinfo
            info(epoch+1).innerinfo = innerinfo;
        end
        
        % Print output
        if options.verbosity > 0
            if break_inneriter == 0
                fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, epoch, cost, gradnorm, stepsize);
            else
                fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e (break at %5d)\n', mode, epoch, cost, gradnorm, stepsize, break_inneriter);
            end
        end
        
        % Stopping criteria
        if gradnorm  <= options.tolgradnorm
            if options.verbosity > 0
                fprintf('\nNorm of gradient smaller than %g.\n',options.tolgradnorm);
            end
            break;
        end
        
    end
    
    info = info(1:epoch+1);
    
    
    % Save the stats per epoch.
    function stats = savestats()
        stats.epoch = epoch;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        stats.grad_cnt = grad_cnt;
        if epoch == 0
            stats.time = 0;
        else
            stats.time = info(epoch).time + elapsed_time;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
    % Save the stats per iteration.
    function innerstats = saveinnerstats()
        innerstats.inneriter = inneriter;
        if inneriter == 0
            innerstats.cost = NaN;
            innerstats.gradnorm = NaN;
            innerstats.time = 0;
        else
            innerstats.cost = cost;
            innerstats.gradnorm = partialgradnorm;
            if inneriter == 1
                innerstats.time = elapsed_time;
            else
                innerstats.time = innerinfo(inneriter-1).time + elapsed_time;
            end
        end
        
    end
    

end


