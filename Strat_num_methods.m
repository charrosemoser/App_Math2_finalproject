%% Stratonovich Numerical Method Comparison for Geometric Brownian Motion (GBM)
% Demonstrates convergence differences in financial SDEs.
clc;
clear all;
close all;

%% Parameters
rng(42); % For reproducibility
mu = -3; % Drift coefficient
sigma = 0.7; % Diffusion coefficient
U0 = 10;% Initial condition
T = 1;% End time
M = 10000;% Number of Monte Carlo paths

N_vec = [16;32;64;128;256;512;1024;2048;4096];

% Preallocation for Strong and Weak Errors
strong_error_Eul = zeros(length(N_vec),1);
strong_error_Mil = zeros(length(N_vec),1);
strong_error_RK = zeros(length(N_vec),1);

% Weak error (mean)
weak_error_Eul = zeros(length(N_vec),1);
weak_error_Mil = zeros(length(N_vec),1);
weak_error_RK = zeros(length(N_vec),1);

for jj = 1:length(N_vec)
    N = N_vec(jj);       % Number of time steps
    dt = T / N;     % Time step size
    
    % Preallocate
    t = linspace(0, T, N+1);
    U_true = zeros(M, N+1);
    u_Eul = zeros(M, N+1);
    u_Mil = zeros(M, N+1);
    u_RK = zeros(M, N+1);
    
    % Monte Carlo Simulation
    for j = 1:M
        dW = sqrt(dt) * randn(1, N); % Wiener increments
        
        % True solution
        W = cumsum(dW);
        U_true(j, :) = U0 * exp((mu)*t + sigma*[0, W]);
        
        % Stratonovich Euler-Maruyama
        u_Eul(j, 1) = U0;
        for i = 1:N
            u_bar = u_Eul(j, i) + mu*u_Eul(j, i)*dt + sigma*u_Eul(j, i)*dW(i);
            u_Eul(j, i+1) = u_Eul(j, i) + 0.5*(mu*u_Eul(j, i)+ mu*u_bar)*dt +...
                0.5*(sigma*u_Eul(j, i) + sigma*u_bar)*dW(i);
        end
        
        % Stratonovich Milstein
        u_Mil(j, 1) = U0;
        for i = 1:N
            u_Mil(j, i+1) = u_Mil(j, i) + mu*u_Mil(j, i)*dt + ...
               sigma*u_Mil(j, i)*dW(i) + 0.5*sigma^2*u_Mil(j, i)*dW(i)^2;
        end
        
        % Stratonovich Runge Kutta
        u_RK(j,1) = U0;
        for i = 1:N
            w  = u_RK(j,i) + (mu * u_RK(j,i) + 0.5*sigma^2* u_RK(j,i))*dt + sigma * u_RK(j,i)*dW(i);
            wp = u_RK(j,i) + (mu * u_RK(j,i)+ 0.5*sigma^2* u_RK(j,i))*dt + sigma * u_RK(j,i)*sqrt(dt);
            wm = u_RK(j,i) + (mu * u_RK(j,i)+ 0.5*sigma^2* u_RK(j,i))*dt - sigma * u_RK(j,i)*sqrt(dt);
            u_RK(j,i+1) = u_RK(j,i) ...
                + 0.5*(mu * u_RK(j,i)+ 0.5*sigma^2* u_RK(j,i) + mu*w + 0.5*sigma^2*w)*dt ...
                + 0.25*(sigma*wp + sigma*wm + 2*sigma * u_RK(j,i))*dW(i) ...
                + 0.25*(sigma*wp - sigma*wm)*(dW(i)^2-dt) / sqrt(dt);
        end


    end
    % Compute Strong and Weak Errors
    % Strong error (pathwise)
    strong_error_Eul(jj) = mean(abs(u_Eul(:, end) - U_true(:, end)));
    strong_error_Mil(jj) = mean(abs(u_Mil(:, end) - U_true(:, end)));
    strong_error_RK(jj) = mean(abs(u_RK(:, end) - U_true(:, end)));
    
    % Weak error (mean)
    weak_error_Eul(jj) = abs(mean(u_Eul(:, end).^2) - mean(U_true(:, end).^2));
    weak_error_Mil(jj) = abs(mean(u_Mil(:, end).^2) - mean(U_true(:, end).^2));
    weak_error_RK(jj) = abs(mean(u_RK(:, end).^2) - mean(U_true(:, end).^2));
end

%% Plot Results
figure
subplot(1,2,1)
hold on
plot(log(N_vec), log(strong_error_Eul), 'b--', 'LineWidth', 2, 'DisplayName', 'Euler-Heun')
plot(log(N_vec), log(strong_error_Mil), 'r-.', 'LineWidth', 2, 'DisplayName', 'Milstein')
plot(log(N_vec), log(strong_error_RK), 'k:', 'LineWidth', 2, 'DisplayName', 'Runge Kutta')
title('Strong Error Comparison')
xlabel('log(N)')
ylabel('log(E)')
set(gca, 'FontSize', 14);
legend
grid on

subplot(1,2,2)
hold on
plot(log(N_vec), log(weak_error_Eul), 'b--', 'LineWidth', 2, 'DisplayName', 'Euler-Heun')
plot(log(N_vec), log(weak_error_Mil), 'r-.', 'LineWidth', 2, 'DisplayName', 'Milstein')
plot(log(N_vec), log(weak_error_RK), 'k:', 'LineWidth', 2, 'DisplayName', 'Runge Kutta')
title('Weak Error Comparison')
xlabel('log(N)')
ylabel('log(E)')
set(gca, 'FontSize', 14);
legend
grid on

% Compute the strong convergence order
fprintf('Strong Convergence Rates:')
fprintf('Euler:')
compute_convergence_order(N_vec,strong_error_Eul)
fprintf('Milstein:')
compute_convergence_order(N_vec,strong_error_Mil)
fprintf('Runge Kutta:')
compute_convergence_order(N_vec,strong_error_RK)

% Compute the weak convergence order
fprintf('Weak Convergence Rates:')
fprintf('Euler:')
compute_convergence_order(N_vec,weak_error_Eul)
fprintf('Milstein:')
compute_convergence_order(N_vec,weak_error_Mil)
fprintf('Runge Kutta:')
compute_convergence_order(N_vec,weak_error_RK)


function p = compute_convergence_order(N_list, error_list)

% Ensure column vectors
N_list = N_list(:);
error_list = error_list(:);

%find dt_list from N_list
dt_list = N_list.^(-1);

% Take logs
log_dt = log(dt_list);
log_error = log(error_list);

% Perform linear regression (least squares fit)
coeffs = polyfit(log_dt, log_error, 1);

% Convergence rate is the slope
p = coeffs(1);

% Optional: display result nicely
fprintf('Estimated order of convergence: %.4f\n', p);

end
