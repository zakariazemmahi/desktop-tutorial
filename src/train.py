
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.collector import get_crypto_data
from src.features.preprocessing import (
    preprocess_data, create_sequences, scale_data, split_data, get_features_list
)
from src.models.cnn_bilstm import build_cnn_bilstm_model, train_model, predict_future_price
from src.utils.visualization import (
    plot_predictions, plot_future_predictions, calculate_metrics,
    print_metrics, plot_training_history
)

def main(args):
    """
    Fonction principale pour l'entraînement et l'évaluation du modèle.

    Args:
        args (argparse.Namespace): Arguments de ligne de commande
    """
    # Création des répertoires nécessaires
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Récupération des données
    print("Récupération des données...")
    data_dict = get_crypto_data(symbols=['BTC', 'ETH'], days=args.days)

    if 'BTC' not in data_dict or 'ETH' not in data_dict:
        print("Erreur: Impossible de récupérer les données BTC et ETH.")
        return

    # Prétraitement des données
    print("Prétraitement des données...")
    data = preprocess_data(data_dict['BTC'], data_dict['ETH'])

    # Sauvegarde des données prétraitées
    data.to_csv('data/preprocessed_data.csv', index=False)
    print(f"✓ Données prétraitées sauvegardées dans 'data/preprocessed_data.csv'")

    # Normalisation des données
    data, feature_scaler, btc_scaler = scale_data(data)

    # Création des séquences
    print(f"Création des séquences (longueur = {args.seq_length})...")
    X, y = create_sequences(data, seq_length=args.seq_length)

    # Division des données
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    print(f"Dimensions des données: ")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Validation: {X_val.shape}, {y_val.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")

    # Construction du modèle
    print("Construction du modèle...")
    model = build_cnn_bilstm_model(args.seq_length, features=len(get_features_list()))
    model.summary()

    # Entraînement du modèle
    print("Entraînement du modèle...")
    history, model = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size
    )

    # Évaluation du modèle
    print("Évaluation du modèle...")
    y_pred = model.predict(X_test)

    # Dénormalisation des résultats
    y_test_btc = btc_scaler.inverse_transform(y_test)
    y_pred_btc = btc_scaler.inverse_transform(y_pred)

    # Calcul des métriques
    metrics = calculate_metrics(y_test_btc, y_pred_btc)
    print_metrics(metrics, "Métriques avant correction: ")

    # Correction du biais
    bias = metrics['Bias']
    y_pred_btc_corrected = y_pred_btc + bias

    # Recalcul des métriques après correction
    metrics_corrected = calculate_metrics(y_test_btc, y_pred_btc_corrected)
    print_metrics(metrics_corrected, "Métriques après correction: ")

    # Visualisation des résultats
    test_dates = data['time'].iloc[-len(y_test):].values
    plot_predictions(
        test_dates, y_test_btc, y_pred_btc, y_pred_btc_corrected,
        save_path='results/test_predictions.png'
    )
    print("✓ Graphique des prédictions sauvegardé dans 'results/test_predictions.png'")

    # Visualisation de l'historique d'entraînement
    plot_training_history(history, save_path='results/training_history.png')
    print("✓ Graphique de l'historique d'entraînement sauvegardé dans 'results/training_history.png'")

    # Sauvegarde du modèle
    model_path = os.path.join('models', args.model_name)
    model.save(model_path)
    print(f"✓ Modèle sauvegardé sous '{model_path}'")

    # Prédiction des prix futurs
    if args.predict_future:
        print(f"Prédiction pour les {args.days_ahead} prochains jours...")
        last_sequence = X[-1]
        future_predictions = predict_future_price(
            model, feature_scaler, btc_scaler, bias, last_sequence,
            days_ahead=args.days_ahead
        )

        # Affichage des prédictions
        print("\nPrévisions des prix futurs :")
        start_date = data['time'].iloc[-1] + timedelta(days=1)
        for i, price in enumerate(future_predictions, 1):
            pred_date = start_date + timedelta(days=i-1)
            print(f"{pred_date.strftime('%Y-%m-%d')}: {price:.2f} USD")

        # Visualisation des prédictions futures
        plot_future_predictions(
            start_date, future_predictions,
            save_path='results/future_predictions.png'
        )
        print("✓ Graphique des prédictions futures sauvegardé dans 'results/future_predictions.png'")

        # Sauvegarde des prédictions futures
        pred_df = pd.DataFrame({
            'date': [start_date + timedelta(days=i) for i in range(len(future_predictions))],
            'predicted_price': future_predictions
        })
        pred_df.to_csv('results/future_predictions.csv', index=False)
        print("✓ Prédictions futures sauvegardées dans 'results/future_predictions.csv'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraînement du modèle de prédiction BTC/ETH')

    parser.add_argument('--days', type=int, default=730,
                        help='Nombre de jours de données historiques à récupérer')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Longueur des séquences temporelles')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion des données pour l\'entraînement')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proportion des données pour la validation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Nombre maximal d\'époques pour l\'entraînement')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille des batchs pour l\'entraînement')
    parser.add_argument('--model_name', type=str, default='model_lstm_bitcoin_eth.h5',
                        help='Nom du fichier pour sauvegarder le modèle')
    parser.add_argument('--predict_future', action='store_true',
                        help='Prédire les prix futurs')
    parser.add_argument('--days_ahead', type=int, default=30,
                        help='Nombre de jours à prédire')

    args = parser.parse_args()
    main(args)
