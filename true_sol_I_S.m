%% Plot Ito and Strat True Solutions

clc;
clear all;
close all;

%% Parameters
rng(42); % For reproducibility
mu = -3; % Drift coefficient
sigma = 0.7; % Diffusion coefficient
U0 = 10; % Initial condition
T = 1; % End time
M = 10000; % Number of Monte Carlo paths
N = 1000; % Number of time steps
dt = T / N; % Time step size

% Preallocate
t = linspace(0, T, N+1);
U_true_I = zeros(M, N+1);
U_true_S = zeros(M, N+1);

% Monte Carlo Simulation
for j = 1:M
    dW = sqrt(dt) * randn(1, N); % Wiener increments
    
    % True solution 
    W = cumsum(dW);
    U_true_S(j, :) = U0 * exp((mu)*t + sigma*[0, W]);
    U_true_I(j, :) = U0 * exp((mu - 0.5*sigma^2)*t + sigma*[0, W]);
end

%% Plot Results
figure
subplot(1,2,1)
hold on
plot(0:dt:T, U_true_I(1,:), 'b', 'LineWidth', 2)
title('(a) One Realization U_I')
xlabel('t')
ylabel('U_I')
set(gca, 'FontSize', 14);
grid on

subplot(1,2,2)
hold on
plot(0:dt:T, mean(U_true_I), 'r', 'LineWidth', 2)
title('(b) Mean of U_I')
xlabel('t')
ylabel('U_I')
set(gca, 'FontSize', 14);
grid on

figure
subplot(1,2,1)
hold on
plot(0:dt:T, U_true_S(1,:), 'b', 'LineWidth', 2)
title('(a) One Realization U_S')
xlabel('t')
ylabel('U_S')
set(gca, 'FontSize', 14);
grid on

subplot(1,2,2)
hold on
plot(0:dt:T, mean(U_true_S), 'r', 'LineWidth', 2)
title('(b) Mean of U_S')
xlabel('t')
ylabel('U_S')
set(gca, 'FontSize', 14);
grid on

