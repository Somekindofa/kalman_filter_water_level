clear all
clc
%% Partie 2 : Estimation d'un niveau d'eau dans un réservoir
clc
% % 2.1 : Estimation d'une constante
load("reservoir1.mat");
K = 0.1;
A = 1;  % Matrice de transition
C = 1;  % Matrice d'observation
Q_v = 0;  % Covariance du bruit de modèle d'état (car `v_n` est un bruit blanc)
Q_b = K;  % Covariance du bruit de mesure
z1 = 0;  % Estimation initiale
P1 = 0.3;  % Covariance initiale de l'erreur

z_est       = zeros(size(y));
P_n         = P1;
P_n_list    = zeros(size(y));
z1_list     = zeros(size(y));
z_old = z1;
mean_v = zeros(size(y));
steps_before_conv = 0;
epsilon = 1e-1;


for n = 1:length(y)
    [G_n, P_n]  = update_filter(A, C, Q_v, Q_b, P_n);
    P_n_list(n) = P_n;
    z_est(n)    = predict(A, C, -G_n, z1, y(n));
    conv_check = abs(z_est(n)-z_old);
    z_old = z_est(n);
    z1          = z_est(n);  % Mettre à jour l'estimation
    if conv_check < epsilon
        mean_v(n)=z_est(n);
    else
        steps_before_conv = steps_before_conv + 1;
    end
    z1_list(n) = z1;
end

% Calculer l'erreur entre la valeur prédite z_est et la valeur réelle y
prediction_error = y - z_est;
% Calculer l'erreur entre l'estimation z1_list et la valeur réelle y
estimation_error = y - z1_list;

fprintf("The convergence is attained with a converged value of %.3f meters \n", sum(mean_v)/nnz(mean_v))
fprintf("On converge à partir de %d iterations\n", steps_before_conv)


figure;
plot(1:length(y), y, 'b');
hold on
plot(1:length(y), z1_list, 'r');

title('Estimation of Water Level');
xlabel('Time Step');
ylabel('Level Surface');
legend('Actual Level', 'Estimated Level');

figure;
plot(1:length(y), P_n_list, 'g')
title("Matrice de covariance de l'erreur du modèle \epsilon_n");
xlabel('Time Step');
ylabel("Variance de l'erreur");
legend('P_n');

%% 2.1 - Choix de K et de P1
clc;
load('reservoir1.mat');

K_values = linspace(0.001, 0.02, 10);
P1_values = linspace(0.001, 0.1, 6);
A = 1;  % Matrice de transition
C = 1;  % Matrice d'observation
Q_v = 0;  % Covariance du bruit de modèle d'état (car `v_n` est un bruit blanc)

figure;

for i = 1:numel(P1_values)
    P1 = P1_values(i);
    
    subplot(2, 3, i);  % Créez un sous-graphique pour chaque combinaison K-P1
    
    for j = 1:numel(K_values)
        K = K_values(j);

        z_est = zeros(size(y));
        P_n = P1;
        z1_list = zeros(size(y));

        for n = 1:length(y)
            [G_n, P_n] = update_filter(A, C, Q_v, K, P_n);
            z_est(n) = predict(A, C, -G_n, z1, y(n));
            z1 = z_est(n);  % Mettre à jour l'estimation
            z1_list(n) = z1;
        end

        % Plot the results for this combination of K and P1
        plot(1:length(y), z1_list, 'LineWidth', 1.5);
        hold on
    end

    % Légende pour chaque courbe (K values)
    legend(arrayfun(@(k) ['K = ', num2str(k)], K_values, 'UniformOutput', false));
    
    title(['P1 = ', num2str(P1)]);
    xlabel('Time Step');
    ylabel('Level Surface');
    grid on;  % Afficher une grille
end



%% Estimation d'une augmentation constante
clc;
clear all;
load('reservoir2.mat');

K   = 0.1;
A   = [1, 1; 0, 1];             % Matrice de transition
C   = [1, 0];                   % Matrice d'observation (seule la position est observable)
Q_v = [0, 0; 0, 0];             % Covariance du bruit de modèle d'état (nulle)
Q_b = K;                        % Covariance du bruit de mesure
z1  = [0; 0];                   % Estimation initiale de [z_n; v_n]
P1  = [1, 1e-5; 1e-5, 1];       % Covariance initiale

z_estimated = cell(length(y), 1); 
z1_list     = cell(length(y), 1);
P_n         = P1;
step = 0;
epsilon = 1e-3;
filling_rate_estimates = zeros(length(y), 1);

for n = 1:length(y)
    Pn_old = P_n(1,1);
    [G_n, P_n] = update_filter(A, C, Q_v, Q_b, P_n);
    Pn_new = P_n(1,1);

    z_estimated{n} = predict(A, C, -G_n, z1, y(n));
    z1 = z_estimated{n};  % Mettre à jour l'estimation pour l'itération suivante
    if n>1 && abs(Pn_new - Pn_old)>epsilon
        step = step+1;
    end
end

fprintf("A partir de %d itérations on obtient une convergence de l'algorithme\n", step)
vertical_height = cellfun(@(x) x(1), z_estimated);
velocity        = cellfun(@(x) x(2), z_estimated);

% Tracer le niveau d'eau réel
figure;
plot(1:length(y), y, 'b');
title('Actual Water Level');
xlabel('Time Step');
ylabel('Water Level');

% Tracer le niveau d'eau estimé
hold on
plot(1:length(y), vertical_height, 'r');
title('Estimated Water Level');
xlabel('Time Step');
ylabel('Water Level');
legend("Hauteur réelle", 'Hauteur estimée')

figure;
plot(1:length(y), velocity, 'g');
title('Estimated Velocity');
xlabel('Time Step');
ylabel('Velocity');


%% PARTIE 3
% 1) voir Préparation
% 2)

clc;
clear all;

% Paramètres
sigma_a = 1;
sigma_m = sqrt(10);
dt = 0.2;

% Initialisations
A = [1, dt; 0, 1];
C = [1, 0];
Q_v = [dt^4/4, dt^3/2; dt^3/2, dt^2] * sigma_a^2;
Q_b = sigma_m^2;

% Simulation pour observer la convergence de P_n
n_iterations = 1000;
P_n_list = zeros(2, 2, n_iterations);

% Initialisations
z_n = [0; 0];
P_n = eye(2);

for n = 1:n_iterations
    % Prédiction
    [G_n, P_n] = update_filter(A, C, Q_v, Q_b, P_n);

    % Stockage de P_n
    P_n_list(:, :, n) = P_n;
end

% Visualisation de l'évolution de P_n
figure;
subplot(2, 2, 1);
plot(squeeze(P_n_list(1, 1, :)));
title('P_{n}(1,1)');

subplot(2, 2, 2);
plot(squeeze(P_n_list(1, 2, :)));
title('P_{n}(1,2)');

subplot(2, 2, 3);
plot(squeeze(P_n_list(2, 1, :)));
title('P_{n}(2,1)');

subplot(2, 2, 4);
plot(squeeze(P_n_list(2, 2, :)));
title('P_{n}(2,2)');

%% 3)
clc;
clear all;

% Charger les données du fichier cible.mat
load('cible.mat');

%% Suivi de la cible pour sigma_a et sigma_m fixés
% Paramètres du filtre de Kalman
sigma_a = 2;
sigma_m = sqrt(10);
delta_t = 0.2;

% Matrices du modèle
A = [1, delta_t; 0, 1];
C = [1, 0];

% Covariances des bruits
Q_v = [1/4 * delta_t^4, 1/2 * delta_t^3; 1/2 * delta_t^3, delta_t^2] * sigma_a^2;
Q_b = sigma_m^2;

% Estimation initiale
z1 = [0; 0];
P1 = [1, 0; 0, 1];

% Initialiser les variables pour stocker les résultats
z_estimated = cell(length(y), 1);

% Initialiser la matrice de covariance pour stocker l'évolution
P_n_list = zeros(2, 2, length(y));
P_n = P1;

% Boucle d'estimation
for n = 1:length(y)
    % Mise à jour du filtre de Kalman
    [G_n, P_n] = update_filter(A, C, Q_v, Q_b, P_n);

    % Prédiction de l'état suivant
    z_estimated{n} = predict(A, C, -G_n, z1, y(n));

    % Mettre à jour l'estimation pour l'itération suivante
    z1 = z_estimated{n};

    % Stocker l'évolution de la matrice de covariance
    P_n_list(:, :, n) = P_n;
end

% Convertir les cellules en tableau
z_estimated = cell2mat(z_estimated.');

% Comparer l'estimation avec les vraies valeurs dans x
figure;
subplot(2, 1, 1);
plot(1:length(y), x(1, :), 'b', 1:length(y), z_estimated(1, :), 'r');
title('Estimation vs Vraie valeur - Position');
legend('Vraie valeur', 'Estimation');
xlabel('Temps');

subplot(2, 1, 2);
plot(1:length(y), x(2, :), 'b', 1:length(y), z_estimated(2, :), 'r');
title('Estimation vs Vraie valeur - Vitesse');
legend('Vraie valeur', 'Estimation');
xlabel('Temps');

%% Suivi de la cible pour plusieurs sigma_a
sigma_m = sqrt(10);
delta_t = 0.2;
load('cible.mat');
sigma_a_values = [0.5, 1, 2, 4];
figure;

% Boucle sur différentes valeurs de sigma_a
for sigma_a_index = 1:length(sigma_a_values)
    sigma_a = sigma_a_values(sigma_a_index);

    A = [1, delta_t; 0, 1];
    C = [1, 0];

    Q_v = [1/4 * delta_t^4, 1/2 * delta_t^3; 1/2 * delta_t^3, delta_t^2] * sigma_a^2;
    Q_b = sigma_m;

    z1 = [0; 0];
    P1 = [1, 0; 0, 1];

    z_estimated = cell(length(y), 1);

    P_n_list = zeros(2, 2, length(y));
    P_n = P1;

    for n = 1:length(y)
        [G_n, P_n] = update_filter(A, C, Q_v, Q_b, P_n);
        z_estimated{n} = predict(A, C, -G_n, z1, y(n));
        z1 = z_estimated{n};
        P_n_list(:, :, n) = P_n;
    end


    z_estimated = cell2mat(z_estimated.');


    subplot(length(sigma_a_values), 2, 2 * sigma_a_index - 1);
    plot(1:length(y), x(1, :), 'b', 1:length(y), z_estimated(1, :), 'r');
    title(['Position, sigma_a = ', num2str(sigma_a)]);
    legend('Vraie valeur', 'Estimation');
    xlabel('Temps');

    subplot(length(sigma_a_values), 2, 2 * sigma_a_index);
    plot(1:length(y), x(2, :), 'b', 1:length(y), z_estimated(2, :), 'r');
    title(['Vitesse, sigma_a = ', num2str(sigma_a)]);
    legend('Vraie valeur', 'Estimation');
    xlabel('Temps');
end

%% Suivi de la cible pour plusieurs sigma_m
% Paramètres du filtre de Kalman
sigma_a = 1;
delta_t = 0.2;
load('cible.mat');
A = [1, delta_t; 0, 1];
C = [1, 0];
z1 = [0; 0];
P1 = [1, 0; 0, 1];

sigma_m_values = linspace(0.1, sqrt(10), 5);

figure;
for i = 1:length(sigma_m_values)
    sigma_m = sigma_m_values(i);
    Q_b = sigma_m;
    z_estimated = cell(length(y), 1);
    P_n_list = zeros(2, 2, length(y));
    P_n = P1;
    for n = 1:length(y)

        [G_n, P_n] = update_filter(A, C, Q_v, Q_b, P_n);

        z_estimated{n} = predict(A, C, -G_n, z1, y(n));

        z1 = z_estimated{n};

        P_n_list(:, :, n) = P_n;
    end


    z_estimated_mat = cell2mat(z_estimated.');


    subplot(length(sigma_m_values), 2, 2*i - 1);
    plot(1:length(y), x(1, :), 'b', 1:length(y), z_estimated_mat(1, :), 'r');
    title(['Position (Sigma_m = ', num2str(sigma_m), ')']);
    legend('Vraie valeur', 'Estimation');
    xlabel('Temps');

    subplot(length(sigma_m_values), 2, 2*i);
    plot(1:length(y), x(2, :), 'b', 1:length(y), z_estimated_mat(2, :), 'r');
    title(['Vitesse (Sigma_m = ', num2str(sigma_m), ')']);
    legend('Vraie valeur', 'Estimation');
    xlabel('Temps');
end


