function M = sympositivedefinitefactory_mod(n)

    symm = @(X) .5*(X+X');
    
    M.name = @() sprintf('New symmetric positive definite geometry of %dx%d matrices', n, n);
    
    M.dim = @() n*(n+1)/2;
    
	% Helpers to avoid computing full matrices simply to extract their trace
	vec     = @(A) A(:);
	trinner = @(A, B) vec(A')'*vec(B);  % = trace(A*B)
	trnorm  = @(A) sqrt(trinner(A, A)); % = sqrt(trace(A^2))
	
    % Choice of the metric on the orthonormal space is motivated by the
    % symmetry present in the space. The metric on the positive definite
    % cone is its natural bi-invariant metric.
	% The result is equal to: trace( (X\eta) * (X\zeta) )
    %M.inner = @(X, eta, zeta) trinner(X\eta, X\zeta);
    M.inner = @innerprod_calc;
    function ip = innerprod_calc(X, eta, zeta)
        eta_mat = reshape(eta, n, n);
        zeta_mat = reshape(zeta, n, n);
        ip = trinner(X\eta_mat, X\zeta_mat);
    end
    
    
    % Notice that X\eta is *not* symmetric in general.
	% The result is equal to: sqrt(trace((X\eta)^2))
    % There should be no need to take the real part, but rounding errors
    % may cause a small imaginary part to appear, so we discard it.
    M.norm = @(X, eta) real(trnorm(X\eta));
    
    % Same here: X\Y is not symmetric in general.
    % Same remark about taking the real part.
    M.dist = @(X, Y) real(trnorm(real(logm(X\Y))));
    
    
    M.typicaldist = @() sqrt(n*(n+1)/2);
    
    
    M.egrad2rgrad = @egrad2rgrad;
    function eta = egrad2rgrad(X, eta)
        eta = X*symm(eta)*X;
    end
    
    
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        % Directional derivatives of the Riemannian gradient
        Hess = X*symm(ehess)*X + 2*symm(eta*symm(egrad)*X);
        
        % Correction factor for the non-constant metric
        Hess = Hess - symm(eta*symm(egrad)*X);
    end
    
    
    M.proj = @(X, eta) symm(eta);
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Y = X + t * eta + 0.5 * t * eta * pinv(X) * t * eta;
        Y = 0.5 * real(Y + Y');
    end    

    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        % The symm() and real() calls are mathematically not necessary but
        % are numerically necessary.
        Y = symm(X*real(expm(X\(t*eta))));
    end
    
    M.log = @logarithm;
    function H = logarithm(X, Y)
        % Same remark regarding the calls to symm() and real().
        H = symm(X*real(logm(X\Y)));
    end
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    % Generate a random symmetric positive definite matrix following a
    % certain distribution. The particular choice of a distribution is of
    % course arbitrary, and specific applications might require different
    % ones.
    M.rand = @random;
    function X = random()
        D = diag(1+rand(n, 1));
        [Q, R] = qr(randn(n)); %#ok<NASGU>
        X = Q*D*Q';
    end
    
    % Generate a uniformly random unit-norm tangent vector at X.
    M.randvec = @randomvec;
    function eta = randomvec(X)
        eta = symm(randn(n));
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(X) zeros(n);
    
    % Poor man's vector transport: exploit the fact that all tangent spaces
    % are the set of symmetric matrices, so that the identity is a sort of
    % vector transport. It may perform poorly if the origin and target (X1
    % and X2) are far apart though. This should not be the case for typical
    % optimization algorithms, which perform small steps.
    %M.transp = @(X1, X2, eta) eta;
    M.paratransp = @vector_transport;
    M.transp = @vector_transport;
    
    function zeta = vector_transport(X, H, eta)
        %   vec_trans(X,N,E) returns the vector transport of tangent vector E (eta) at 
        %   the point X in the direction of tangent vector N(H).         
        expconstruct=expm(H/X/2);
        zeta=expconstruct * eta * expconstruct';
        
    end    
    
    % For reference, a proper vector transport is given here, following
    % work by Sra and Hosseini: "Conic geometric optimisation on the
    % manifold of positive definite matrices", to appear in SIAM J. Optim.
    % in 2015; also available here: http://arxiv.org/abs/1312.1039
    % This will not be used by default. To force the use of this transport,
    % execute "M.transp = M.parallel_transp;" on your M returned by the
    % present factory.
    M.paralleltransp = @parallel_transport;
    function zeta = parallel_transport(X, Y, eta)
        E = sqrtm((Y/X));
        zeta = E*eta*E';
    end


    %%
    % isometric vector transport with locking condition.
    % Input and output are assumed extrinsic representations by
    % following work: "A Riemannian Limited-Memory BFGS Algorithm for 
    % Computing the Matrix Geometric Mean" by X.Yuan, We. Huang, P.-A.
    % ABsil and K.A. Gallivan.
    % The following codes are modified from the one obtained from 
    % http://www.math.fsu.edu/~whuang2/papers/ARLBACMGM.htm.
    M.transp_locking = @Tranv_locking_ext;
    function [output, fd, x, y] = Tranv_locking_ext(x, fd, y, fv) 
        
        if ~norm(fd)
            output = fv;
            return;
        end

        d = obtain_intrep(x, fd);
        v = obtain_intrep(x, fv);    
        eps1 = d;
        Tee = Tranv_R_new(x, fd, y, fd);
        Tee_intr = obtain_intrep(y, Tee);
        beta = sqrt(inpro_intrinsic(x, d, d) / inpro_intrinsic(y, Tee_intr, Tee_intr));
        M.beta = beta;
        eps2 = beta * Tee_intr;
        nu1 = 2 * eps1;
        nu2 = - eps1 - eps2;

        output = v - 2 * inpro_intrinsic(y, nu1, v) / inpro_intrinsic(y, nu1, nu1) * nu1;
        output = output - 2 * inpro_intrinsic(y, nu2, output) / inpro_intrinsic(y, nu2, nu2) * nu2;
        output = obtain_extrep(y, output);
    end


    % inverse of vector transport with locking condition
    function [output, fd, x, y] = invTranv_locking_ext(x, fd, y, fv) 

        d = obtain_intrep(x, fd);
        v = obtain_intrep(y, fv);    
        eps1 = d;
        Tee = Tranv_R_new(x, fd, y, fd);
        Tee_intr = obtain_intrep(y, Tee);
        beta = sqrt(inpro_intrinsic(x, d, d) / inpro_intrinsic(y, Tee_intr, Tee_intr));
        eps2 = beta * Tee_intr;
        nu1 = 2 * eps1;
        nu2 = - eps1 - eps2;    

        output = v - 2 * inpro_intrinsic(y, nu2, v) / inpro_intrinsic(y, nu2, nu2) * nu2;
        output = output - 2 * inpro_intrinsic(y, nu1, output) / inpro_intrinsic(y, nu1, nu1) * nu1;
        output = obtain_extrep(x, output);    
    end


    % obtain extrinsic representation
    function output = obtain_extrep(x, v) 

        L = chol(x,'lower');
        n = size(x, 1);
        omega = tril(ones(n, n), -1);
        indx = find(omega);
        omega(indx) = v(1 : 0.5 * n * (n - 1)) / sqrt(2);
        output = omega + omega' + diag(v(1 + 0.5 * n * (n - 1) : end));

        output = L * output * L';
    end


    % obtain intrinsic representation
    function output = obtain_intrep(x, eta)
        
        L = chol(x,'lower');
        invL = inv(L);

        n = size(eta, 1);
        indx = find(tril(ones(n, n), -1));
        eta_new = invL * eta * invL';
        output = zeros(0.5 * n * (n + 1), 1);
        output(1 : 0.5 * n * (n - 1)) = sqrt(2) * eta_new(indx);
        output(1 + 0.5 * n * (n - 1) : end) = diag(eta_new);
    end

    function [output, fd, x, y] = Tranv_R_new(x, fd, y, fv) % Differentiated vector transport
        temp = 0.5 * fv * pinv(x) * fd;
        output = fv + temp + temp';
    end

    % intrinsic representation for inner product
    function output = inpro_intrinsic(x, v1, v2)
        output = v1' * v2;
    end

    M.obtain_beta = @get_beta;
    function output = get_beta(x)
        output = M.beta;
    end


    
    % vec and mat are not isometries, because of the unusual inner metric.
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, n);
    M.vecmatareisometries = @() false;
       
    
end
