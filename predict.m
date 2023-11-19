function z_pred = predict(A, C, G_n, z_n, x_n) 
    z_pred = A * z_n; % Prédiction de l'état suivant
    e_n = C * z_pred - x_n; % Calcul de l'innovation
    z_pred = z_pred + A * G_n * e_n; % Mise à jour de l'état
end
