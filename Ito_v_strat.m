%% Ito vs Stratonovich Simulation
% Demonstrates convergence differences in a financial SDE.

clc;
clear all;
close all;

%% Parameters
rng(42); % For reproducibility
mu = 0.75;% Drift coefficient
sigma = 0.3;% Diffusion coefficient
U0 = 307.65;% Initial condition
T = 2; % End time
N = 1000;% Number of time steps
dt = T / N;% Time step size
M = 10000;% Number of Monte Carlo paths

% Preallocate
t = linspace(0, T, N+1);
U_true = zeros(M, N+1);
U_ito = zeros(M, N+1);
U_strat_uncorrected = zeros(M, N+1);
U_strat_corrected = zeros(M, N+1);

%% Monte Carlo Simulation
for j = 1:M
    dW = sqrt(dt) * randn(1, N); % Wiener increments
    
    % True solution 
    W = cumsum(dW);
    U_true(j, :) = U0 * exp((mu - 0.5*sigma^2)*t + sigma*[0, W]);
    
    % Itô Euler-Maruyama
    U_ito(j, 1) = U0;
    for i = 1:N
        U_ito(j, i+1) = U_ito(j, i) + mu*U_ito(j, i)*dt + sigma*U_ito(j, i)*dW(i);
    end
    
    % Stratonovich Heun's (uncorrected)
    U_strat_uncorrected(j, 1) = U0;
        for i = 1:N
            u_bar = U_strat_uncorrected(j, i) + mu*U_strat_uncorrected(j, i)*dt + sigma*U_strat_uncorrected(j, i)*dW(i);
            U_strat_uncorrected(j, i+1) = U_strat_uncorrected(j, i) + 0.5*(mu*U_strat_uncorrected(j, i)+ mu*u_bar)*dt +...
                0.5*(sigma*U_strat_uncorrected(j, i) + sigma*u_bar)*dW(i);
        end
    
    % Stratonovich Heun's 
    U_strat_corrected(j, 1) = U0;
    for i = 1:N
        U_bar = U_strat_corrected(j, i) + (mu - 0.5*sigma^2)*U_strat_corrected(j, i)*dt + sigma*U_strat_corrected(j, i)*dW(i);
        U_strat_corrected(j, i+1) = U_strat_corrected(j, i) + ...
            0.5*((mu - 0.5*sigma^2)*U_strat_corrected(j, i) + (mu - 0.5*sigma^2)*U_bar)*dt + ...
            0.5*sigma*(U_strat_corrected(j, i) + U_bar)*dW(i);
    end
end

%% Compute Strong and Weak Errors
% Strong error (pathwise)
strong_error_ito = mean(abs(U_ito(:, end) - U_true(:, end)));
strong_error_strat_uncorrected = mean(abs(U_strat_uncorrected(:, end) - U_true(:, end)));
strong_error_strat_corrected = mean(abs(U_strat_corrected(:, end) - U_true(:, end)));

% Weak error (mean)
weak_error_ito = abs(mean(U_ito(:, end)) - mean(U_true(:, end)));
weak_error_strat_uncorrected = abs(mean(U_strat_uncorrected(:, end)) - mean(U_true(:, end)));
weak_error_strat_corrected = abs(mean(U_strat_corrected(:, end)) - mean(U_true(:, end)));

%% Plot Results
figure
subplot(1, 2, 1)
hold on
plot(t, mean(U_true), 'k-', 'LineWidth', 2, 'DisplayName', 'True Solution')
plot(t, mean(U_ito), 'b--', 'LineWidth', 2, 'DisplayName', 'Itô Euler-Maruyama')
plot(t, mean(U_strat_uncorrected), 'r-.', 'LineWidth', 2, 'DisplayName', 'Stratonovich (Uncorrected)')
plot(t, mean(U_strat_corrected), 'g:', 'LineWidth', 2, 'DisplayName', 'Stratonovich (Corrected)')
xlabel('Time')
ylabel('Mean of X_t')
title('Comparison of Mean Paths')
legend
set(gca, 'FontSize', 14);
grid on

subplot(1, 2, 2)
hold on
plot(t, U_true(1,:), 'k-', 'LineWidth', 2, 'DisplayName', 'True Solution')
plot(t, U_ito(1,:), 'b--', 'LineWidth', 2, 'DisplayName', 'Itô Euler-Maruyama')
plot(t, U_strat_uncorrected(1,:), 'r-.', 'LineWidth', 2, 'DisplayName', 'Stratonovich (Uncorrected)')
plot(t, U_strat_corrected(1,:), 'g:', 'LineWidth', 2, 'DisplayName', 'Stratonovich (Corrected)')
xlabel('Time')
ylabel('One Trajectory of U')
title('Comparison of Mean Paths')
set(gca, 'FontSize', 14);
grid on

figure
bar([strong_error_ito, strong_error_strat_uncorrected, strong_error_strat_corrected; ...
     weak_error_ito, weak_error_strat_uncorrected, weak_error_strat_corrected]);
set(gca, 'XTickLabel', {'Strong Error', 'Weak Error'})
legend('Ito', 'Strat (Uncorrected)', 'Strat (Corrected)')
title('Strong and Weak Errors at t=T')
ylabel('Error Magnitude')
set(gca, 'FontSize', 14)
grid on

%% Display Errors
fprintf('Strong Errors:\n')
fprintf('  Itô: %.4f\n', strong_error_ito)
fprintf('  Stratonovich (Uncorrected): %.4f\n', strong_error_strat_uncorrected)
fprintf('  Stratonovich (Corrected): %.4f\n\n', strong_error_strat_corrected)

fprintf('Weak Errors:\n');
fprintf('  Itô: %.4f\n', weak_error_ito);
fprintf('  Stratonovich (Uncorrected): %.4f\n', weak_error_strat_uncorrected)
fprintf('  Stratonovich (Corrected): %.4f\n', weak_error_strat_corrected)