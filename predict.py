#!/usr/bin/env python
"""
Script principal pour faire des prédictions avec un modèle existant.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from src.data.collector import get_crypto_data
from src.features.preprocessing import preprocess_data, create_sequences, scale_data
from src.models.cnn_bilstm import predict_future_price

def main(args):
    """
    Fonction principale pour charger un modèle et faire des prédictions.

    Args:
        args (argparse.Namespace): Arguments de ligne de commande
    """
    # Vérifier si le modèle existe
    if not os.path.exists(args.model_path):
        print(f"Erreur: Le modèle {args.model_path} n'existe pas.")
        return

    # Récupération des données récentes
    print("Récupération des données récentes...")
    data_dict = get_crypto_data(symbols=['BTC', 'ETH'], days=args.days)

    if 'BTC' not in data_dict or 'ETH' not in data_dict:
        print("Erreur: Impossible de récupérer les données BTC et ETH.")
        return

    # Prétraitement des données
    print("Prétraitement des données...")
    data = preprocess_data(data_dict['BTC'], data_dict['ETH'])

    # Normalisation des données
    data, feature_scaler, btc_scaler = scale_data(data)

    # Création des séquences
    seq_length = args.seq_length
    X, y = create_sequences(data, seq_length=seq_length)

    if len(X) == 0:
        print(f"Erreur: Pas assez de données pour créer des séquences de longueur {seq_length}.")
        return

    # Chargement du modèle
    print(f"Chargement du modèle depuis {args.model_path}...")
    model = load_model(args.model_path)

    # Calcul du biais de correction
    y_pred = model.predict(X)
    y_real = btc_scaler.inverse_transform(y)
    y_pred_denorm = btc_scaler.inverse_transform(y_pred)
    bias = np.mean(y_real - y_pred_denorm)
    print(f"Biais de correction calculé: {bias:.2f}")

    # Prédiction des prix futurs
    print(f"Prédiction pour les {args.days_ahead} prochains jours...")
    last_sequence = X[-1]  # Dernière séquence connue
    future_predictions = predict_future_price(
        model, feature_scaler, btc_scaler, bias, last_sequence,
        days_ahead=args.days_ahead
    )

    # Création du répertoire des résultats si nécessaire
    os.makedirs('results', exist_ok=True)

    # Affichage des prédictions
    print("\nPrévisions des prix futurs du Bitcoin:")
    start_date = data['time'].iloc[-1] + timedelta(days=1)
    dates = []
    for i, price in enumerate(future_predictions, 1):
        pred_date = start_date + timedelta(days=i-1)
        dates.append(pred_date)
        print(f"{pred_date.strftime('%Y-%m-%d')}: {price:.2f} USD")

    # Visualisation des prédictions futures
    plt.figure(figsize=(12, 6))
    plt.plot(dates, future_predictions, marker='o', linestyle='-', color='green')
    plt.title(f'Prédiction du prix Bitcoin pour les {args.days_ahead} prochains jours')
    plt.xlabel('Date')
    plt.ylabel('Prix (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Sauvegarde du graphique
    plt.savefig('results/btc_prediction.png')
    print(f"✓ Graphique des prédictions sauvegardé dans 'results/btc_prediction.png'")

    # Sauvegarde des prédictions dans un CSV
    pred_df = pd.DataFrame({
        'date': dates,
        'predicted_price': future_predictions
    })
    pred_df.to_csv('results/btc_prediction.csv', index=False)
    print(f"✓ Prédictions sauvegardées dans 'results/btc_prediction.csv'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prédiction du prix Bitcoin basée sur Ethereum')

    parser.add_argument('--model_path', type=str, default='models/model_lstm_bitcoin_eth.h5',
                      help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--days', type=int, default=60,
                      help='Nombre de jours de données historiques à récupérer')
    parser.add_argument('--seq_length', type=int, default=30,
                      help='Longueur des séquences temporelles')
    parser.add_argument('--days_ahead', type=int, default=30,
                      help='Nombre de jours à prédire')

    args = parser.parse_args()
    main(args)
