function [ f, pJ ] = Phi_BAD( U, param)

% Backward AD of the nonlinear function in the objective

% Forward pass
N  = length(U);
x0 = param.x0;
h  = param.T/N;
q  = param.q;

X    = zeros(N+1,1);
X(1) = x0;
for k = 1:N
    X(k+1) = X(k) + h*( (1 - X(k))*X(k) + U(k));
end

f = q*X(end).^2;

% Backward AD. only one output, so we only need to seed once
Ubar = zeros(N,1);
Xbar = zeros(N+1,1);
fbar = 1;               % this is our seed vector

% recursively obtain bar quantities
Xbar(end) = Xbar(end) + 2*q*X(end)*fbar;

for k = 1:N
    Xbar(N+1-k) = Xbar(N+1-k) + (1 + h - 2*h*X(N+1-k))*Xbar(N+2-k);
    Ubar(N+1-k) = Ubar(N+1-k) + h*Xbar(N+2-k);
end

pJ = Ubar';   % the bar quantities of the inputs are our jacobian

end

