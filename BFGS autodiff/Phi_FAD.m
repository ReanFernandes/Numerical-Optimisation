function [ f, J ] = Phi_FAD( U, param )

% Forward AD of the nonlinear function in the objective

% Forward pass
N  = length(U);
x0 = param.x0;
h  = param.T/N;
q  = param.q;

X = zeros(N+1,1);
X(1) = x0;

for k = 1:N
    X(k+1) = X(k) + h*( (1 - X(k))*X(k) + U(k));
end

f = q*X(end).^2;

% obtain Jacobian
P = eye(N);
J = zeros(1,N);

% we need to seed once for every column of the jacobian
for ii = 1:N
    p = P(:,ii);                    % seed vector
    Udot    = p;                    % seed input variable
    Xdot    = zeros(N+1,1);
    % obtain all dot quantities
    for k = 1:N
    	dphidx = 1 + h * (1 - 2 * X(k));
    	dphidu = h;
        Xdot(k+1) = dphidx * Xdot(k) + dphidu * Udot(k);
    end
    
    % the dot quantity of output f is our jacobian entry
    J(ii) = 2*q*X(end)*Xdot(end);
    
end

end

