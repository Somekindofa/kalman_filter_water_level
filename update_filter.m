function [G_n, P_n] = update_filter(A, C, Q_v, Q_b, P_n)
    G_n = (P_n * C')/(Q_b + C*P_n*C'); % Calcul du gain de Kalman
    P_n_inter = P_n - G_n  * C * P_n; % Covariance de l'erreur d'observation intermediaire
    P_n = A * P_n_inter * A' + Q_v; % update de la matrice de covariance de l'erreur d'observation
end
