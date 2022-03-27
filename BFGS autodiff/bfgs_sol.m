%% BFGS method
clear all;
close all;
clc;


% GRADIENT = 'forward_AD';
GRADIENT = 'backward_AD';
% GRADIENT = 'i_trick';
% GRADIENT = 'finite_differences';
% GRADIENT = 'casadi';

param.N  = 200;        % number of discretization steps
param.x0 = 2;           % initial condition on state
param.T  = 5;           % terminal time
param.q  = 50;        % weight on terminal state
h = param.T/param.N;


obj = @(U) U.'*U + Phi(U,param);

U = zeros(param.N,1);   % initial controls
B = eye(param.N);       % initial Hessian

% Printing header: iterate number, gradient norm, objective value, step norm, stepsize
fprintf('It.\t | ||grad_f||| f\t\t | ||dvar||\t | t  \n');

% Get the objective value and the gradient
switch GRADIENT    
    case 'finite_differences'
        [f, J] = finite_difference(@Phi, U, param);
    case 'i_trick'
        [f, J] = i_trick(@Phi, U, param);
    case 'backward_AD'
        [f, J] = Phi_BAD(U, param);
    case 'forward_AD'
        [f, J] = Phi_FAD(U, param);
        
    case 'casadi'
        import casadi.*
        Uvar  = SX.sym('u',param.N);
        phi_expr = param.x0;
        for i=1:param.N
            phi_expr = phi_expr + h * ((1 - phi_expr) * phi_expr + Uvar(i) );
        end
        phi_expr = param.q * phi_expr^2;
        Phi_func = Function('Phi',{Uvar},{phi_expr});
        J_func   = Function('J',{Uvar},{jacobian(phi_expr,Uvar)});
        f = full(Phi_func(U));
        J = full(J_func(U));
end

J = J + 2*U.';
f = f + sum(U.^2);

maxit = 300;
tol   = 1e-3;
tlog  = 0;


% Iterative loop
for k = 1 : maxit
    % Obtaining search direction
    dx = -B \ J';
    
    % We check Armijo's condition to find a step length
    t     = 1.0;    % Initial step length
    beta  = 0.8;    % Shrinking factor
    gamma = 0.1;    % Minimal decrease requirement
    
    U_new = U + t * dx; % Candidate for the next step

    % Iterate on Armijo's condition
    while obj(U_new) > f + gamma * t * J * dx
        t = beta * t;
        U_new = U + t * dx;
    end
    %var_new = var + dvar;
    
    tic
    switch GRADIENT
        case 'finite_differences'
            [f_new, J_new] = finite_difference(@Phi , U_new, param);
        case 'i_trick'
            [f_new, J_new] = i_trick(@Phi , U_new, param);
        case 'backward_AD'
            [f_new, J_new] = Phi_BAD(U_new, param);
        case 'forward_AD'
            [f_new, J_new] =  Phi_FAD(U_new, param);
        case 'casadi'
            f_new = full(Phi_func(U_new));
            J_new = full(J_func(U_new));
    end
    tlog = tlog + toc;
    
    J_new = J_new + 2*U_new.';
    f_new = f_new + sum(U_new.^2);
    
    % Updating Hessian according to BFGS formula
    s = U_new - U;
    z = J_new' - J';
    B = B - B * ( s * s') * B / (s' * B * s) + z * z' / (s' * z);
    
    % Updating variables
    U = U_new;
    J = J_new;
    f = f_new;

    % Every 10 iterations print the header again
    if mod(k,10) == 0
        fprintf('\n');
        fprintf('It.\t | ||grad_f||| f\t\t | ||dvar||\t | t  \n');
    end
    
    % Print some useful information
    fprintf('%d\t | %8.5f\t | %8.5f\t | %8.5f\t | %f \n', k, norm(J), f, norm(dx), t);
    
    if norm(J) < tol
        disp('Convergence achieved.');
        break
    end

end

t = 0:h:param.T;

% plot controls
stairs(t,[U; U(end)],'r')
hold on

% reconstruct states
X = zeros(param.N+1,1);
X(1) = param.x0;

for i = 1:param.N
    X(i+1) = X(i) + h*((1-X(i))*X(i)+U(i));
end

% plot trajectory
plot(t,X)

legend('control','state')
title('Optimal solution')
xlabel('Time')

disp(' ')
disp('TOTAL TIME SPENT IN DERIVATIVES')
disp(' ')
disp(tlog)