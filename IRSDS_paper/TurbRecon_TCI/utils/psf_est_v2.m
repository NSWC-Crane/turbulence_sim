function z = psf_est_v2(A, b, lambda, rho, mu, sig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Point spread function estimation function for version 2. 
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[m, n] = size(A);
Atb = A'*b;

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

[L, U] = factor(A, rho);

for k = 1:MAX_ITER
    
    % x-update
    q = Atb + rho*(z - u); 
    if( m >= n )
       x = U \ (L \ q);
    else 
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end

    zold = z;
    x_hat = x;
    z = shrinkage((x_hat + u - mu), lambda .* 1./sig./rho);

    u = u + (x_hat - z);

    r_norm = norm(x - z);
    s_norm = norm(-rho*(z - zold));
    
    eps_pri  = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    eps_dual = sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
    
    if r_norm < eps_pri && s_norm < eps_dual
         break;
    end

end

end


function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L, U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n ) 
       L = chol( A'*A + rho*speye(n), 'lower' );
    else          
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end
    
    L = sparse(L);
    U = sparse(L');
end
