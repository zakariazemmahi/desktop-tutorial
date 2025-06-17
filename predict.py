import numpy as np
import pandas as pd
from datetime import datetime
import time
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from data.collector import collect_data_crypto_compare
from features.preprocessing import preprocess_data
from models.model import create_sequences

def load_and_predict(seq_length=32):
    print("🔁 Chargement du modèle...")
    model = load_model("best_best_model.h5")
    print("✅ Modèle chargé avec succès.")

    # Récupération des données récentes
    today = datetime.today()
    end_timestamp = int(time.mktime(today.timetuple()))
    start_timestamp = 1614556800  # 1 mars 2021

    btc_data = collect_data_crypto_compare('BTC', start_timestamp, end_timestamp)
    eth_data = collect_data_crypto_compare('ETH', start_timestamp, end_timestamp)

    if btc_data is None or eth_data is None:
        print("❌ Échec de la récupération des données.")
        return

    df = preprocess_data(btc_data, eth_data)

    eth_scaler = MinMaxScaler()
    btc_scaler = MinMaxScaler()

    eth_scaled = eth_scaler.fit_transform(df[['close_eth']])
    btc_scaled = btc_scaler.fit_transform(df[['close_btc']])

    X, y = create_sequences(eth_scaled, btc_scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 2))

    last_sequence = X[-1].reshape(1, seq_length, 2)
    predicted_scaled = model.predict(last_sequence)
    predicted_price = btc_scaler.inverse_transform(predicted_scaled)

    print(f"📈 Dernière prédiction du prix BTC : {predicted_price[0][0]:.2f} USD")

    return predicted_price[0][0]

if __name__ == "__main__":
    load_and_predict()
